from torchvision import transforms


def get_tta_transforms(input_size=224):
    """
    Generate asymmetric clean/weak/strong views for teacher-student TTA.

    The normalization constants follow the CLIP convention used by common
    medical vision-language foundation models. Replace them when using a
    backbone with different preprocessing requirements.
    """
    norm_mean = [0.48145466, 0.4578275, 0.40821073]
    norm_std = [0.26862954, 0.26130258, 0.27577711]
    
    # Base Transformation (Clean) - Strictly for evaluation and basic inference
    base_aug = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    # Weak Augmentation (Teacher Input) - Provides reliable pseudo-targets
    weak_aug = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    # Strong Augmentation (Student Input) - Enforces robustness against severe perturbations
    strong_aug = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), 
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)), 
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    return base_aug, weak_aug, strong_aug
