from keras.models import load_model
model=load_model("/content/drive/My Drive/LATEST2/Gantrained.h5") 
model.summary()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        "/content/drive/My Drive/Latest2/inputldpts/",
        target_size=(104, 104),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)