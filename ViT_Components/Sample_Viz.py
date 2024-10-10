# Class to visualize a random image from dataset in order to know if data is loaded properly

import matplotlib.pyplot as plt

class Sample_Viz():
    def __init__(self):
        pass
    
    def visualize_sample(self, train_dataloader, class_names):
        # Get a batch of images
        image_batch, label_batch = next(iter(train_dataloader))

        # Get a single image from the batch
        image, label = image_batch[0], label_batch[0]

        # View the batch shapes
        print(image.shape, label)

        # Plot image with matplotlib
        plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
        plt.title(class_names[label])
        plt.axis(False)

        #tensor(0) means first class 0 defaut