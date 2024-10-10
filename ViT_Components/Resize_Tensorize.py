from torchvision import transforms

# Class for Resize and tensor transformation pipeline
class Resize_Tensorize():
    def __init__(self):
        self.IMG_SIZE = 224

    def create_transforms(self):
        # Create a transform pipeline
        transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        print(f"Applied transforms: {transform}")
        return transform
    
