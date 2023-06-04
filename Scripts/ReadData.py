
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------Prepare dataset
def Data_Generator(Label_file_dir, Images_dir, Batch_size=50, image_size=(224,224) ):

    Data_df = pd.read_csv (Label_file_dir)
    Fine_labels = Data_df['Finding']
    main_labels = Fine_labels.where((Fine_labels=='esophagitis-a') | (Fine_labels=='esophagitis-b-d'), 'Barrett' )
    main_labels = main_labels.where((Fine_labels=='barretts-short-segment') | (Fine_labels=='barretts'), 'Inflammation' )
    Data_df['main_labels'] = main_labels
    Data_df = Data_df.sample(frac=1)

    Data_df['Images_SubPath'] =  Data_df['Finding'] + '/' + Data_df['File'] +'.jpg'


    train_datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2, horizontal_flip = True, vertical_flip=True) #

    train_generator = train_datagen.flow_from_dataframe(
                                            dataframe=Data_df,
                                            directory=Images_dir,
                                            x_col='Images_SubPath',
                                            y_col='main_labels',
                                            batch_size=Batch_size,
                                            subset='training',
                                            class_mode='binary',
                                            target_size=(image_size[0], image_size[1]),
                                            shuffle=True
                                            )
    test_generator = train_datagen.flow_from_dataframe(
                                            dataframe=Data_df,
                                            directory=Images_dir,
                                            x_col='Images_SubPath',
                                            y_col='main_labels',
                                            batch_size=Batch_size,
                                            subset='validation',
                                            class_mode='binary',
                                            target_size=(image_size[0], image_size[1]),
                                            shuffle=False
                                            )

    # ------ Statistics and class weights
    print(Fine_labels.value_counts())
    print(main_labels.value_counts())

    Barrett_Sample_num = main_labels.value_counts()['Barrett']
    Inflammation_Sample_num = main_labels.value_counts()['Inflammation']
    total_Sample_num = Barrett_Sample_num + Inflammation_Sample_num

    weight_for_Barrett = (1 / Barrett_Sample_num) * (total_Sample_num / 2.0)  # Positive samoples
    weight_for_Inflammation = (1 / Inflammation_Sample_num) * (total_Sample_num / 2.0)  # Negative Samples



    return train_generator, test_generator, weight_for_Barrett, weight_for_Inflammation