import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_num_instances(translate: dict, data_path: str):
    '''
    Plot the number of instances for each class in the dataset.
    Args:
        translate: a dictionary with the italian and english names of the classes
        data_path: the path to the dataset
    '''
    # Add labels and statistics to two different lists
    classes = translate.values()
    num_instances = []
    for animal_folder in translate.keys():
        num_instances.append(len(os.listdir(os.path.join(data_path, animal_folder))))
    
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.bar(classes, num_instances, color='skyblue')
    # Add title and labels to the plot
    plt.title('Number of instances')
    plt.ylabel('Counts')
    
    plt.subplot(2, 1, 2)
    plt.title('Percentage of instances')
    plt.pie(num_instances, labels=classes, autopct='%1.1f%%')

    # Show the plot
    plt.tight_layout(pad = 5.0)
    plt.show()
    
    
def plot_image_sizes(translate: dict, data_path: str):
    '''
    Plot the sizes of the images in the dataset.
    Args:
        translate: a dictionary with the italian and english names of the classes
        data_path: the path to the dataset
    '''
    # create subplots
    fig, ax = plt.subplots(5, 2, figsize=(10, 20))
    fig.suptitle('Image sizes')

    heights = {}
    widths = {}
    for animal_folder in translate.keys():
        heights[animal_folder] = []
        widths[animal_folder] = []
        for image in os.listdir(os.path.join(data_path, animal_folder)):
            img = cv2.imread(os.path.join(data_path, animal_folder, image))
            heights[animal_folder].append(img.shape[0])
            widths[animal_folder].append(img.shape[1])

    for i, animal in enumerate(translate.keys()):
        # Scatter plot of the heights and widths of the images
        ax[i // 2, i % 2].scatter(heights[animal], widths[animal], color='skyblue', s=5, alpha=0.7)
        ax[i // 2, i % 2].set_title(translate[animal])
        ax[i // 2, i % 2].set_xlabel('Height')
        ax[i // 2, i % 2].set_ylabel('Width')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
        

def show_random_images(translate: dict, data_path: str):
    '''
    Show 5 random images for each class in the dataset.
    Args:
        translate: a dictionary with the italian and english names of the classes
        data_path: the path to the dataset
    '''
    # Show 5 images for each class
    fig, ax = plt.subplots(10, 5, figsize=(10, 30))
    for i, animal_folder in enumerate(translate.keys()):
        for j, image in enumerate(np.random.choice(os.listdir(os.path.join(data_path, animal_folder)), 5)):
            img = cv2.imread(os.path.join(data_path, animal_folder, image))
            ax[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[i, j].set_title(translate[animal_folder])
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.show()
        
