import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionConstraintLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_center=1.0, lambda_soft_cosine=1.0):
        super(EmotionConstraintLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_center = lambda_center
        self.lambda_soft_cosine = lambda_soft_cosine
        
        # 初始化类中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels, logits, similarity_matrix):
        # 1. 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, labels)

        # 2. 计算中心损失
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        center_loss = torch.sum((features - centers_batch) ** 2) / 2.0 / batch_size

        # 3. 计算软余弦相似度损失
        soft_cosine_loss = 0.0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    cosine_similarity = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0))
                    soft_cosine_loss += similarity_matrix[labels[i], labels[j]] * F.relu(1.0 - cosine_similarity)
        
        soft_cosine_loss /= (batch_size * (batch_size - 1))

        # 4. 总损失
        total_loss = ce_loss + self.lambda_center * center_loss + self.lambda_soft_cosine * soft_cosine_loss

        return total_loss

