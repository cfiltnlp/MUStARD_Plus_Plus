import torch
import torch.nn as nn
import numpy as np
import random

'''Comment these to randomize the model training'''
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

"""This file contains all the models we have used for -
1. Multiclass Implicit Emotion Classification
2. Multiclass Explicit Emotion Classification
3. Multimodal Sarcasm Detection
    3.1 for MUStARD
    3.2 for MUStARD++


Class names and Variable names are self explanatory
Before setting the values to input embedding  we first sort the modality name in descending order.
(VTA) in order to remove randomness in the model

Parameters:
      input_embedding_A:
            Takes the input dimension of first modality
      input_embedding_B:
            Takes the input dimension of second modality
      input_embedding_C:
            Takes the input dimension of third modality
      shared_embedding:
            This is the dimension size to which we have to project all modality, to have equal dimension input from each input modality
      projection_embedding:
            This is the intermediate dimension size to which project our shared embedding to calculate attention
      dropout: 
            Parameter to pass dropout (to be hyper-tuned)


we assign "num_classes" variable depending upon the task
for example,
    a. num_classes=5 (5) (Multiclass Implicit Emotion Classification)
    b. num_classes=9 (Multiclass Explicit Emotion Classification)
    c. num_classes=2 (Multimodal Sarcasm Detection)

Output Layer = Softmax Layer
"""

audio_embedding_size = 291 #Modify the embedding sizes based on audio feature extraction chosen:Ex: 314, 319
class Speaker_Independent_Triple_Mode_with_Context(nn.Module):
    def __init__(self, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Triple_Mode_with_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_context_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)
        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calculates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD, feE, feF):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + \
            self.attention(feA, feC) + \
            self.attention(feA, feD) + \
            self.attention(feA, feE) + \
            self.attention(feA, feF)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, uC, cC):
        """Args:
                uA:
                    Utterance Video
                uB:
                    Utterance Text
                uC:
                    Utterance Audio
                cA:
                    Context Video
                cB:
                    Context Text
                cC:
                    Context Audio

            Returns:
                probability of emotion classes
                (
                    Since we have used Cross-entropy as loss function,
                    Therefore we have not used softmax here because Cross-entropy perform Softmax while calculating loss
                    While evaluation we have to perform softmax explicitly
                )
        """
        """ Feature Projection, in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_context = self.norm_C_context(
            nn.functional.relu(self.C_context_share(cC)))
        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        # Feature Modulation
        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Independent_Dual_Mode_with_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Dual_Mode_with_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calculates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + \
            self.attention(feA, feC) + \
            self.attention(feA, feD)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_B_context, shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Independent_Single_Mode_with_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Single_Mode_with_Context, self).__init__()

        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.context_share = nn.Linear(
            self.input_embedding, self.shared_embedding)
        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calculates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA):
        """ Feature Projection, in order to make all feature of same dimension"""

        shared_context = self.norm_context(
            nn.functional.relu(self.context_share(cA)))
        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_context)

        input = updated_shared

        return self.pred_module(updated_shared)

################################################################################################################################################################################################################


class Speaker_Independent_Triple_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Triple_Mode_without_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calcuates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + self.attention(feA, feC)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, uB,  uC):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,   shared_C_utterance,  shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance,   shared_A_utterance,  shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,   shared_A_utterance,  shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Independent_Dual_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Independent_Dual_Mode_without_Context, self).__init__()

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA,  uB):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,  shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,  shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Independent_Single_Mode_without_Context(nn.Module):
    def __init__(self, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.2, num_classes=5):
        super(Speaker_Independent_Single_Mode_without_Context, self).__init__()
        print("No. of classes:",num_classes)
        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            # nn.Linear(3*self.shared_embedding, 2*self.shared_embedding),
            # nn.BatchNorm1d(2*self.shared_embedding),
            # nn.ReLU(),
            # nn.Linear(2*self.shared_embedding, self.shared_embedding),
            # nn.BatchNorm1d(self.shared_embedding),
            # nn.ReLU(),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calcuates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_utterance)

        input = updated_shared

        return self.pred_module(updated_shared)
################################################################################################################################################################################################################


class Speaker_Dependent_Triple_Mode_with_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Triple_Mode_with_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_context_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)
        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+3*self.shared_embedding,
                      2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)

        )

    def attention(self, featureA, featureB):
        """ This method takes two features and calculates the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD, feE, feF):
        """ This method calculates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + self.attention(feA, feC) + self.attention(
            feA, feD) + self.attention(feA, feE) + self.attention(feA, feF)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, uC, cC, speaker_embedding):
        """Args:
                uA:
                    Utterance Video
                uB:
                    Utterance Text
                uC:
                    Utterance Audio
                cA:
                    Context Video
                cB:
                    Context Text
                cC:
                    Context Audio

            Returns:
                probability of emotion classes
                (
                    Since we have used Crossentropy as loss function,
                    Therefore we have not used softmax here because Crossentropy perform Softmax while calculating loss
                    While evaluation we have to perform softmax explicitly
                )
        """
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_context = self.norm_C_context(
            nn.functional.relu(self.C_context_share(cC)))
        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Dual_Mode_with_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Dual_Mode_with_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_context_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_context_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+2*self.shared_embedding,
                      self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC, feD):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + self.attention(feA,
                                                          feC) + self.attention(feA, feD)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_B_context, shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Single_Mode_with_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Single_Mode_with_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.context_share = nn.Linear(
            self.input_embedding, self.shared_embedding)
        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, cA, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_context = self.norm_context(
            nn.functional.relu(self.context_share(cA)))
        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_context)

        input = torch.cat((updated_shared, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Triple_Mode_without_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=2048, input_embedding_B=1024, input_embedding_C=audio_embedding_size, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Triple_Mode_without_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.input_embedding_C = input_embedding_C
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.C_utterance_share = nn.Linear(
            self.input_embedding_C, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+3*self.shared_embedding,
                      2*self.shared_embedding),
            nn.BatchNorm1d(2*self.shared_embedding),
            nn.ReLU(),
            nn.Linear(2*self.shared_embedding, self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB, feC):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB) + self.attention(feA, feC)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, uB,  uC, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,   shared_C_utterance,  shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance,   shared_A_utterance,  shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,   shared_A_utterance,  shared_C_utterance)

        temp = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((temp, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Dual_Mode_without_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, input_embedding_B=2048, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Dual_Mode_without_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding_A = input_embedding_A
        self.input_embedding_B = input_embedding_B
        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.A_utterance_share = nn.Linear(
            self.input_embedding_A, self.shared_embedding)

        self.B_utterance_share = nn.Linear(
            self.input_embedding_B, self.shared_embedding)

        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+2*self.shared_embedding,
                      self.shared_embedding),
            nn.BatchNorm1d(self.shared_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_embedding,  512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA,  uB, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance,  shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance,  shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.pred_module(input)

################################################################################################################################################################################################################


class Speaker_Dependent_Single_Mode_without_Context(nn.Module):
    def __init__(self, n_speaker=24, input_embedding_A=1024, shared_embedding=1024, projection_embedding=512, dropout=0.5, num_classes=5):
        super(Speaker_Dependent_Single_Mode_without_Context, self).__init__()

        self.n_speaker = n_speaker

        self.input_embedding = input_embedding_A

        self.shared_embedding = shared_embedding
        self.projection_embedding = projection_embedding
        self.num_classes = num_classes
        self.dropout = dropout

        self.utterance_share = nn.Linear(
            self.input_embedding, self.shared_embedding)

        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collabrative_gate_1 = nn.Linear(
            2*self.shared_embedding, self.projection_embedding)
        self.collabrative_gate_2 = nn.Linear(
            self.projection_embedding, self.shared_embedding)

        self.pred_module = nn.Sequential(
            nn.Linear(self.n_speaker+self.shared_embedding, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,  128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,  self.num_classes)
        )

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collabrative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""    
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collabrative_gate_2(input), dim=1)

    def forward(self, uA, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_utterance)

        input = torch.cat((updated_shared, speaker_embedding), dim=1)
        return self.pred_module(input)

################################################################################################################################################################################################################
