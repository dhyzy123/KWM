import torch
from networks.matchbox.ConvASRDecoder import ConvASRDecoderClassification
from networks.matchbox.ConvASREncoder import ConvASREncoder
import torch.nn as nn

class EncDecBaseModel(nn.Module):
    """Encoder decoder Classification models."""

    def __init__(self, num_mels, 
                 final_filter,
                 num_classes):
        super(EncDecBaseModel, self).__init__()

        self.encoder = ConvASREncoder(feat_in = num_mels)
        self.decoder = ConvASRDecoderClassification(feat_in = final_filter, num_classes= num_classes)

    def forward(self, input_signal, input_signal_length=98):
        encoded, encoded_len = self.encoder(audio_signal=input_signal, length=input_signal_length)
        logits = self.decoder(encoder_output=encoded)
        return logits

if __name__=='__main__':
    model = EncDecBaseModel(num_mels= 64, final_filter = 128, num_classes=12)
    test_input = torch.rand([256, 98, 64])
    test_output = model(test_input, 98)

    print(test_output.size())
