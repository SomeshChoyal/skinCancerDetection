import torch
import timm
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Define class labels
CLASS_LABELS = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_LABELS)

# Keep your original model definition
class HybridMHSA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(HybridMHSA, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.mhsa(x, x, x)
        attn_output = self.dropout(attn_output)
        conv_output = self.conv(x.permute(1, 2, 0))
        conv_output = conv_output.permute(2, 0, 1)
        x = attn_output + conv_output
        x = self.norm(x)
        return x

class ModifiedViT(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, dropout=0.1):
        super(ModifiedViT, self).__init__()
        self.base_model = timm.create_model(pretrained_model_name, pretrained=False, drop_path_rate=dropout)

        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                setattr(
                    self.base_model,
                    name,
                    HybridMHSA(
                        dim=module.embed_dim,
                        num_heads=module.num_heads,
                        dropout=dropout
                    )
                )

        self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Load model
def load_model(checkpoint_path="/home/som/Desktop/SkinCancer/checkpoint.pth", device="cpu"):
    model = ModifiedViT('vit_base_patch16_224_in21k', num_classes=len(CLASS_LABELS))
    checkpoint = torch.load(checkpoint_path, map_location=device ,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode
    return model

# Initialize model and move to device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(device=DEVICE)
