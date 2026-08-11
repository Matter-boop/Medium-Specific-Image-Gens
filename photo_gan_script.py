#!/usr/bin/env python
# coding: utf-8

# In[2]:
#Imports
import numpy as np 
import matplotlib.pyplot as plt 
import keras 
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU 
from keras.layers import UpSampling2D, Conv2D 
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam,SGD
from keras.datasets import cifar10
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from IPython.display import Image, display
import torch
import lpips
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
print("GPUs:", tf.config.list_physical_devices('GPU'))
# In[ ]:
#Open ds
import pickle

with open("photo_array.pkl", "rb") as f:
    photo_ds = pickle.load(f)

# In[ ]:
#Image specs
image_shape = (512,512,3) #Images are set at 512x512 pixels, 3 represents the colours channel RGB
latent_dimensions = 100 #Size of noise input for generator


# In[ ]:

def build_generator():
    model = keras.Sequential()

    model.add(Dense(128 * 16 * 16, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((16, 16, 128)))

    # 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(256, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # 64x64
    model.add(UpSampling2D())
    model.add(Conv2D(256, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # 128x128
    model.add(UpSampling2D())
    model.add(Conv2D(256, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # 256x256
    model.add(UpSampling2D())
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # 512x512
    model.add(UpSampling2D())
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(3, 3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)

    return Model(noise, image)


# In[ ]: 
def build_discriminator(): 
    model = Sequential() 

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=image_shape, padding="same")) #halving 32 onwards
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same")) 
    model.add(ZeroPadding2D(padding=((0,1),(0,1)))) 
    #model.add(BatchNormalization(momentum=0.82)) 
    model.add(LeakyReLU(alpha=0.25)) 
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same")) 
    #model.add(BatchNormalization(momentum=0.82)) 
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same")) 
    #model.add(BatchNormalization(momentum=0.8)) 
    model.add(LeakyReLU(alpha=0.25)) 
    model.add(Dropout(0.25)) 
    
    model.add(Flatten()) 
    model.add(Dense(1, activation='sigmoid')) 

    image = Input(shape=image_shape) 
    validity = model(image) 

    return Model(image, validity)


# In[ ]:
#Function for displaying images during training
def display_images(generator): 
    r, c = 4,4
    noise = np.random.normal(0, 1, (r * c,latent_dimensions)) 
    generated_images = generator.predict(noise) 

    generated_images = 0.5 * generated_images + 0.5 #Re-scales images

    fig, axs = plt.subplots(r, c) #Allows multiple images to be plotted
    count = 0
    for i in range(r): 
        for j in range(c): 
            axs[i,j].imshow(generated_images[count, :,:,]) 
            axs[i,j].axis('off') 
            count += 1
    plt.show() 
    plt.close()

# In[ ]:
#function for displaying images at the end of training at full res
def display_images_full_res(generator):

    r, c = 4, 4

    noise = np.random.normal(0, 1, (r * c, latent_dimensions))

    generated_images = generator.predict(noise)

    generated_images = ((generated_images + 1) * 127.5)
    generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)

    image_size = 512

    dpi = 100
    fig_width = (c * image_size) / dpi
    fig_height = (r * image_size) / dpi

    fig, axs = plt.subplots(r, c, figsize=(fig_width, fig_height), dpi=dpi)

    count = 0

    for i in range(r):
        for j in range(c):

            axs[i, j].imshow(generated_images[count])

            axs[i, j].axis('off')

            count += 1

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0
    )

    #plt.savefig("generated_grid.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
# In[ ]: 
#calculate FID score
#InceptionV3 is a pre-trained model made by google that converts images into feature vectors
inception = InceptionV3(
    include_top=False,
    pooling='avg',
    input_shape=(299, 299, 3)
)
#InceptionV3 needs images sized 299x299 so we resize
def process(paths):
    resized = tf.image.resize(paths, (299, 299))
    return resized.numpy()

#FID function
def calculate_fid(real_images, fake_images):
    #because of tahn activation we need to make sure our fake images are 0-255
    fake_images = (fake_images + 1) * 127.5

    #convert images to 299x299
    real_images = process(real_images)
    fake_images = process(fake_images)

    #Preprocess images for InceptionV3 (normalise)
    real_images = preprocess_input(real_images.astype('float32'))
    fake_images = preprocess_input(fake_images.astype('float32'))

    #Get features from InceptionV3
    real_features = inception.predict(real_images)
    fake_features = inception.predict(fake_images)

    #Calculate mean and covariance of features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    mean_diff = mu_real - mu_fake

    covmean = sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return (np.sum(mean_diff ** 2)+ np.trace(sigma_real + sigma_fake - 2 * covmean))

# In[ ]:
#Lpips function

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def prepare_for_lpips(images):
    
    images = images.astype(np.float32)

    #if images are [0,255], convert to [-1,1]
    if images.max() > 1:
        images = (images / 127.5) - 1

    # convert to proper shape (N, C, H, W)
    images = np.transpose(images, (0, 3, 1, 2))

    #convert to torch tensor
    return torch.tensor(images)
# In[ ]:

#Builds dismcriminator for photos

# discriminator = build_discriminator()

discriminator = load_model("discriminator_model_photo_40000.h5")
discriminator.load_weights("discriminator_weights_photo_40000.weights.h5")

discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', 
                    optimizer=Adam(0.00005,beta_1=0.5),#optimizer=Adam(0.00015,beta_1=0.5),
                    metrics=['accuracy']) 

discriminator.trainable = False


# In[ ]:

#Combines model for photo

# generator = build_generator() 

generator = load_model("generator_model_photo_40000.h5")
generator.load_weights("generator_weights_photo_40000.weights.h5")

z = Input(shape=(latent_dimensions,)) 
image = generator(z) 

valid = discriminator(image) 


combined_network = Model(z, valid) 
combined_network.compile(loss='binary_crossentropy', 
                        optimizer=Adam(0.0001,0.5)) #optimizer=Adam(0.00015,beta_1=0.5),


# In[ ]: Training the GAN

SIZE = 512 #Downsizing for storage
X = photo_ds #Array of images in decimal form
print(X.shape)
starter_epoch = 40000
num_epochs = 50001 - starter_epoch
batch_size = 4
display_interval = 100

#best_fid = 9999
losses = []

X = X.astype("float32") / 127.5 - 1.

valid = np.ones((batch_size, 1)) *.9 #smoothes out the labels for discrim as sigmoid activaton can only be max 1
#valid += 0.05 * np.random.random(valid.shape)
fake = np.zeros((batch_size, 1)) 
#fake += 0.05 * np.random.random(fake.shape) 

for epoch in range(num_epochs): 
    #if epoch % 1 == 0: #We're "Nerfing" the discriminator by only training it every other epoch
    #if epoch % 2 == 0:
    index = np.random.randint(0, X.shape[0], batch_size) 
    images = X[index] 

    noise = np.random.normal(0, 1, (batch_size, latent_dimensions)) 
    #generated_images = generator.predict(noise)
    generated_images = generator(noise, training=False)

    discm_loss_real = discriminator.train_on_batch(images, valid) 
    discm_loss_fake = discriminator.train_on_batch(generated_images, fake) 
    discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake) 

    
    genr_loss = combined_network.train_on_batch(noise, valid) 
    
    if epoch % display_interval == 0: 
        display_images(generator)
        print(f"{epoch+starter_epoch} [D loss: {discm_loss[0]:.4f}, acc: {100*discm_loss[1]:.2f}%] [G loss: {genr_loss:.4f}]")

    if epoch % 1000 == 0:
        #real_image = X[:64]
        #fake_image = generated_images

        #real_act = inception(preprocess(real_image)).numpy()
        #fake_act = inception(preprocess(fake_image)).numpy()

        #fid = calculate_fid(real_act, fake_act)
        generator.save("generator_model_photo_fin_" + str(epoch+starter_epoch) + ".h5")
        generator.save_weights("generator_weights_photo_fin_" + str(epoch+starter_epoch) + ".weights.h5")
        discriminator.save("discriminator_model_photo_fin_" + str(epoch+starter_epoch) + ".h5")
        discriminator.save_weights("discriminator_weights_photo_fin_" + str(epoch+starter_epoch) + ".weights.h5")

# In[]:
num_test_images = 200
batch_size = 4
epochs_to_test = [41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000]
#G_losses = []
#D_losses = []
FID_scores = []
SSIM_scores = []
LPIPS_scores = []
for i in range(10):
    epoch = 41000 + i*1000
    generator = load_model("generator_model_photo_fin_" + str(epoch) + ".h5")
    generator.load_weights("generator_weights_photo_fin_" + str(epoch) + ".weights.h5")
    # discriminator = load_model("discriminator_model_photo_fin_" + str(64000 + i*1000) + ".h5")
    # discriminator.load_weights("discriminator_weights_photo_fin_" + str(64000 + i*1000) + ".weights.h5")
    noise = np.random.normal(0, 1, (1, latent_dimensions))

    generated = generator.predict(noise)
    fake_images = []

    for j in range(0, num_test_images, batch_size):

        noise = np.random.normal(0, 1,(batch_size, latent_dimensions))

        batch = generator.predict(noise, verbose=0)

        fake_images.append(batch)

    fake_images = np.concatenate(fake_images, axis=0)
    
    #Calculating scores
    calculated_fid = calculate_fid(photo_ds[:200],fake_images)
    fake_image = (fake_images[0] + 1) * 127.5
    ssim_skimg = ssim(photo_ds[0], fake_image, channel_axis=-1, data_range=255)
    display_images_full_res(generator)
    
    #Append scores to lists for plotting later
    #G_losses.append(f"{genr_loss:.4f}")
    #D_losses.append(f"{discm_loss[0]:.2f}")
    FID_scores.append(calculated_fid)
    SSIM_scores.append(ssim_skimg)
    LPIPS_scores.append(loss_fn_alex(prepare_for_lpips(photo_ds[0:1]), prepare_for_lpips(fake_images[0:1])).item())

    #Scores printed
    #print(f"{epoch} [D loss: {discm_loss[0]:.4f}, acc: {100*discm_loss[1]:.2f}%] [G loss: {genr_loss:.4f}]")
    print(f"FID Score: {FID_scores[i]}")
    print(f"SSIM Score: {SSIM_scores[i]}")
    print(f"LPIPS Score: {LPIPS_scores[i]}")
# In[ ]:
#Plotting scores
# In[ ]:
#Openning pickle
import pickle

with open("photo_scores.pkl", "rb") as f:
    data = pickle.load(f)

FID_scores = data["FID_scores"]
SSIM_scores = data["SSIM_scores"]
LPIPS_scores = data["LPIPS_scores"]
epochs_to_test = data["epochs"]

# In[ ]:
#normalising the scores where 1 is best 0 is worst
FID_scores_norm = [1 - (fid - min(FID_scores)) / (max(FID_scores) - min(FID_scores)) for fid in FID_scores]
SSIM_scores_norm = SSIM_scores #SSIM is already between 0 and 1 where 1 is best
LPIPS_scores_norm = [-(lpips -1) for lpips in LPIPS_scores]

# In[ ]:
#Plotting scores

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(8,10))

axs[0].plot(epochs_to_test, FID_scores_norm, marker='o')
axs[0].set_title("Inverted Normalised FID")
axs[0].set_ylabel("Score")
axs[0].set_xlabel("Epoch")
axs[0].margins(x=0, y=0)
axs[0].grid()

axs[1].plot(epochs_to_test, LPIPS_scores_norm, marker='o')
axs[1].set_title("Inverted  LPIPS")
axs[1].set_ylabel("Score")
axs[1].set_xlabel("Epoch")
axs[1].set_ylim(0, 1)
axs[1].margins(x=0, y=0)
axs[1].grid()

axs[2].plot(epochs_to_test, SSIM_scores_norm, marker='o')
axs[2].set_title("SSIM")
axs[2].set_ylabel("Score")
axs[2].set_xlabel("Epoch")
axs[2].set_ylim(0, 1)
axs[2].margins(x=0, y=0)
axs[2].grid()


plt.tight_layout()
plt.show()

# %%
with open("photo_scores.pkl", "wb") as f:
    pickle.dump({
        "epochs": epochs_to_test,
        #"G_losses": G_losses,
        #"D_losses": D_losses,
        "FID_scores": FID_scores,
        "SSIM_scores": SSIM_scores,
        "LPIPS_scores": LPIPS_scores
    }, f)
# %%
