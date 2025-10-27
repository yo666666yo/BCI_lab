from torch import nn
from MultiDecoderEEG import EEGEncoder, DecoderTemplate

# multi decoder for training tasks
class MultiDecoder4Train(nn.Module):
    def __init__(self,
            # assume our that final goal is 5-classification task, and 2 & 3 are easier tasks for training 
            n_chan, n_cls_list=[2,'''3, 5'''],
            F=128, F_T=64, K_T=3, T=256, L=2, dropout=0.5):
        super(MultiDecoder4Train, self).__init__()
        self.encoder = EEGEncoder(
            n_chan=n_chan,
            F=F,
            F_T=F_T,
            K_T=K_T,
            T=T,
            L=L
        )
        self.decoders = nn.ModuleList()
        for n_cls in n_cls_list:
            self.decoders.append(
                DecoderTemplate(
                    n_cls=n_cls,
                    dropout=dropout
                )
            )

    def forward(self, x):
        x = self.encoder(x)
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(x))
        return outputs
    
# multi decoder for precise control
class MultiDecoder4Control(nn.Module):
    def __init__(self,
            n_chan, n_cls_list=[3, 5], # 3 is for speed/force, 5 is for directions
            F=128, F_T=64, K_T=3, T=256, L=2, dropout=0.5):
        super(MultiDecoder4Control, self).__init__()
        self.encoder = EEGEncoder(
            n_chan=n_chan,
            F=F,
            F_T=F_T,
            K_T=K_T,
            T=T,
            L=L
        )
        self.decoders = nn.ModuleList()
        for n_cls in n_cls_list:
            self.decoders.append(
                DecoderTemplate(
                    n_cls=n_cls,
                    dropout=dropout
                )
            )

    def forward(self, x):
        x = self.encoder(x)
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(x))
        movement = outputs[0] * outputs[1] # movement <- force * direction
        return movement