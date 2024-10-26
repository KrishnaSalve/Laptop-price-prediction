# Laptop Price Predictor.

![Image](https://github.com/user-attachments/assets/fae5c8c4-e04a-43ba-8139-5726e8e5c041)


Welcome to the Laptop Price Prediction project! This project is dedicated to developing a predictive model that can estimate the price of laptops based on their specifications and features. In an ever-evolving laptop market, accurately predicting the value of a laptop can be beneficial for both consumers and sellers. By leveraging data and machine learning techniques, we aim to provide a valuable tool for understanding the factors that influence laptop pricing.

## Description 

In this README, you will find an overview of the project's objectives, the methodology employed, the structure of the code, the technologies used, and guidance on how to get started with the project. Whether you are a data enthusiast, a consumer looking to make an informed purchase, or a seller seeking to set competitive prices, this project aims to provide valuable insights into laptop pricing.

I encourage you to explore the accompanying Jupyter Notebook, where you will find detailed implementation, analysis, and findings from the predictive model. We hope that this project proves to be a useful resource, and we are excited to share our journey of predicting laptop prices using machine learning techniques.

## Overview

This project aims to create a model that can predict the price of a laptop based on various features and specifications. The goal is to help both consumers and sellers in understanding the value of a laptop based on its attributes, ultimately aiding in making informed decisions and setting appropriate prices.


## Data
The project utilizes a dataset containing information about various laptops, including features such as Laptop Company, TypeName, Inches, Screen Resolution, Cpu, Ram, Memory, Gpu, Operating System, Weight and Price. This data will be used to train and test the predictive model.

## Methodology 
The project will employ machine learning techniques, specifically regression algorithms, to build the predictive model. The dataset will be preprocessed to handle missing values, Extract important features from column, encode categorical variables and use column transformer with Pipeline. Various regression algorithms such as linear regression, decision tree regression, or random forest regression will be implemented and evaluated to determine the most accurate model for predicting laptop prices. We will also use Voting Regression and Stacking Regression to get the more accuracte model.

## Model Structure
The project is organized into the following sections:


### Features 
1. Data Collection and Data Cleaning.
- This Laptop dataset contains so many features such as Manufacturing company of Laptop, its Type, screen resolution, screen size, Ram, Memory, Cpu, Operating system, Weight which helps to predict Laptop price. This dataset can be aquired from Kaggle. This dataset contains of 1303 records with 13 columns. 
- Before data processing we have first cleaned data where we had 30 rows with missing rows which we dropped using dropna and have also dropped duplicated values which can introduce biasness and skew the distribution of data.

      # Droping all the Null Values from dataset
      data.dropna(inplace = True)

      # To drop all the duplicate values from dataset.
      data.drop_duplicates(inplace = True)


2. Exploratory Data Analysis.
- We have also done some data analysis to understand correlation between Independent variables and Dependent variable such as Prices.
- To get better understanding of the EDA take a look at Laptop Price Predictor Jupyter Notebook.

3. Data Preprocessing.
- Many features in our dataset such as Screen Resolution, Cpu, Ram, Memory, Operating System, and Weight required Data Preprocessing.
- Ram and Weight column contains Numerical as well as string datatype where Ram contains 'GB' after integer for ex. 8GB and weight column contains 'kg' after float value for ex. 1.36kg. To get Numerical values we have replaced string values with empty spaces.

      # Replacing string values with empty spaces.
      data['Ram'] = data['Ram'].str.replace('GB', '')
      data['Weight'] = data['Weight'].str.replace('kg', '')

      # Converting datatypes of Numeric columns from object to int or float. 
      data['Ram'] = pd.to_numeric(data['Ram'], errors = 'coerce')
      data['Weight'] = pd.to_numeric(data['Weight'], errors = 'coerce')
      data['Inches'] = pd.to_numeric(data['Inches'], errors = 'coerce')

- Screen Resolution column contains so many important features such is screen Touchscreen, is it IPS, or just have resolution_width x resolution_height, To extract this values from columns we have done some data processing and create new columns.

      # Touchscreen Screen Resolution.
      data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)

      # IPS Screen Resolution
      data['IPS'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

      # Resolution Width and Height
      data['Resolution_Width'] = data['ScreenResolution'].str.extract(r'(\d+)x(\d+)')[0]
      data['Resolution_Height'] = data['ScreenResolution'].str.extract(r'(\d+)x(\d+)')[1]

- We have created new feature as 'PPI - Pixel per Inch' which is calculated as, 

      # PPI - Pixels per Inch.
      data['PPI'] = (((data['Resolution_Width']**2) + (data['Resolution_Height']**2))**0.5/data['Inches']).astype('float')


- Similary to screen resolution columns we have also extracted important information from Cpu column.

      # Extracting Cpu Brand.
      data['Cpu_Brand'] = data['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

      # Grouping Cpu Brands.
      def fetch_processor(text):
      if text == "Intel Core i7" or text == "Intel Core i5" or text == "Intel Core i3":
          return text
      elif text.split()[0] == "Intel":
          return "Other Intel Processor"
      elif text.split()[0] == "Samsung":
          return "Samsung Processor"
      else: 
          return "AMD Processor"
      
      data['Cpu_Brand'] = data['Cpu_Brand'].apply(fetch_processor)

- Memory columns Data prrprocessing.

      # Extracting features from Memory column.
      data['Memory'] = data['Memory'].str.replace('\.0', '', regex = True)
      data['Memory'] = data['Memory'].str.replace('GB', '')
      data['Memory'] = data['Memory'].str.replace('TB', '000')
      new = data['Memory'].str.split('+', n = 1, expand = True)
      
      data['first'] = new[0]
      data['first'] = data['first'].str.strip()

      data['second'] = new[1]

      data['layer1HDD'] = data['first'].apply(lambda x: 1 if "HDD" in x else 0)
      data['layer1SSD'] = data['first'].apply(lambda x: 1 if "SSD" in x else 0)
      data['layer1Hybrid'] = data['first'].apply(lambda x: 1 if "Hybrid" in x else 0)
      data['layer1Flash_Storage'] = data['first'].apply(lambda x: 1 if "Flash Storage" in x else 0)


        data['first'] = data['first'].str.extract(r'(\d+)')
        data['second'].fillna('0', inplace = True)

        data['layer2HDD'] = data['second'].apply(lambda x: 1 if "HDD" in x else 0)
        data['layer2SSD'] = data['second'].apply(lambda x: 1 if "SSD" in x else 0)
        data['layer2Hybrid'] = data['second'].apply(lambda x: 1 if "Hybrid" in x else 0)
        data['layer2Flash_Storage'] = data['second'].apply(lambda x: 1 if "Flash Storage" in x else 0)

        data['second'] = data['second'].str.extract(r'(\d+)')

        data['first'] = data['first'].astype(int)
        data['second'] = data['second'].astype(int)

        data['HDD'] = (data['first'] * data['layer1HDD'] + data['second'] * data     ['layer2HDD'])
        data['SSD'] = (data['first'] * data['layer1SSD'] + data['second'] * data['layer2SSD'])
        data['Hybrid'] = (data['first'] * data['layer1Hybrid'] + data['second'] * data['layer2Hybrid'])
        data['Flash_Storage'] = (data['first'] * data['layer1Flash_Storage'] + data['second'] * data['layer2Flash_Storage'])


        data.drop(columns = ['first', 'second', 'layer1HDD', 'layer2HDD', 'layer1SSD', 'layer2SSD', 'layer1Hybrid', 'layer2Hybrid', 'layer1Flash_Storage', 'layer2Flash_Storage'], axis = 1, inplace = True)

for better understanding of above code please go through my Laptop Price Predictor Jupyter Notebook.

- Data preprocessing for Gpu column.

        # Extracting Gpu brand from Gpu column.
        data['Gpu_Brand'] = data['Gpu'].apply(lambda x: x.split()[0])

- Data preprocessing of Operating system.

        # Grouping Various OS 
        def Operating_system(text):
        if text == 'Windows 10' or text == 'Windows 7' or text == 'Windows 10 S':
            return 'Windows'
        elif text == 'macOS' or text == 'Mac OS X':
            return 'Mac'
        else: 
             return 'Others/No OS/Linux'

        # Applying above function to OpSys column
        data['OpSys'] = data['OpSys'].apply(Operating_system)

4. Model Training.
- Before going to Data modelling we have used column transformer, pipeline to first use feature engineering using column transformer and then use various algorithms to get most accurate prices.

        ohe = OneHotEncoder()
        ohe.fit(x[['Company', 'TypeName', 'OpSys', 'Cpu_Brand', 'Gpu_Brand']])

        step1 = ColumnTransformer(transformers = [
            ('col_tfn', OneHotEncoder(categories = ohe.categories_, drop = 'first'), 
            ['Company', 'TypeName', 'OpSys', 'Cpu_Brand', 'Gpu_Brand'])], 
            remainder = 'passthrough')

        step2 = LinearRegression()

        pipe = Pipeline([
            ('step1', step1),
            ('step2', step2)
        ])

        pipe.fit(x_train, y_train)

        ycap = pipe.predict(x_test)
        print('R_score', r2_score(y_test, ycap))
        print('MAE', mean_absolute_error(y_test, ycap))

  similary, we have used various algrithms to train our model also used Voting and Stacking Regression.

5. Model Evaluation.
- Here we got model performances after training the model using various algorithms. Below is the screen shot of all the algorithms we have used with there performance evaluation. Below we have algorithms with their R_score and mean absolute error respectively.
 
  ![Screenshot_20241024_115654](https://github.com/user-attachments/assets/3928cbb2-81e3-43f8-85f5-806d973fdd8a)

Further, I have also used Hyperparameter for some algorithms which were giving use good R score. Please go through my Laptop Price Predictor Jupyter Notebook to understand and read more about Hyperparameter code.


## Technologies Used / Installation
### Prerequisites

- Python 3.8 or above 
- Anaconda Navigator (not necessary)
- Other libraries: Numpy, Matplotlib, xgboost, Streamlit


## Setup

  1. Clone the Repository:

    https://github.com/KrishnaSalve/Laptop-price-prediction.git

  2. Navigate to the project directory in your local system:

    cd Laptop-price-predictor

  3. Install required packages: 

    pip install -r requirements.txt

### Usage 

  1. Run the Streamlit Application:

         streamlit run app.py


  This will start the web application and make it accessible at  http://localhost:5000/

  2. Add Laptop Configuration:

  - Open your web browser and navigate to http://localhost:5000/
  - After you are navigated to your localhost url you will see selectebox options related to laptop specification like Brand, Type, Ram in GB, Weight of the Laptop, Touchscreen, IPS, Screen Size, Screen Resolution, Cpu, HDD(in GB), SSD(in GB), Gpu, OpSys.
  - All the options are selectbox where you have to select the option which are mentioned ar were in our dataset except two specifications Weight of the Laptop and Screen size(Inch of screen) where you have to specify weight and screen size.

3. View the Prediction:
- After you specify the options click on the 'Predict Price' button.
- The predicted price will be shown on the web page as, "The Predicted price of this configuration is Rs. {predicted price}"

4. Interpreting the result:
- Based on the specified configuration your model will predict the price of the Laptop.
- Basic difference between our Notebook model and web page model is they are same but in our Notebook model PPI should be given in model to predict the price.
- PPI are not given with laptop specification but are supposed to calculated using Screen size and Screen Resolution. To know more about PPI calculations look for `app.py` file and Jupyter Notebook for Laptop Price Predictor.


### Result
The Laptop Price Predictor is a machine learning model that predicts the price of a laptop based on various features such as processor type, RAM, storage capacity, screen size, and other specifications.
The model has been evaluated using various validation techniques and has achieved a high level of accuracy with very low mean absolute error and R score of 91% in predicting laptop prices. However, it's important to note that the predictions are estimates and may vary based on market conditions and other factors.

Qualitative Results:

The models performance on the test dataset is as follows:

|Metric |   Value|
|-|-|
|Mean Absolute Error(mae)| 0.15
|R score| 91.1%

### Contributing
We welcome contributions from the community to help improve and expand Laptop price predictor project. If you're interested in contributing, please follow these guidelines:

**Report Issues** : 

If you encounter any bugs, errors, or have suggestions for improvements, please open an issue on the project's GitHub repository. When reporting an issue, provide a clear and detailed description of the problem, including any relevant error messages or screenshots.

**Submit Bug Fixes or Enhancements** : 

If you've identified a bug or have an idea for an enhancement, feel free to submit a pull request. Before starting work, please check the existing issues to ensure that your proposed change hasn't already been addressed.
When submitting a pull request, make sure to:

    1. Fork the repository and create a new branch for your changes.
    2. Clearly describe the problem you're solving and the proposed solution in the pull request description.
    3. Follow the project's coding style and conventions.
    4. Include relevant tests (if applicable) to ensure the stability of your changes.
    5. Update the documentation, including the README file, if necessary.


**Improve Documentation**

If you notice any issues or have suggestions for improving the project's documentation, such as the README file, please submit a pull request with the necessary changes.

**Provide Feedback**

We value your feedback and suggestions for improving the Laptop Price Predictor project. Feel free to share your thoughts, ideas, or use cases by opening an issue or reaching out to the project maintainers.


### Contact

If you have any questions, feedback, or would like to get in touch with the project maintainers, you can reach us through the following channels:

- **Project Maintainer**

  Name : Krishna Salve 

  Email : krishnasalve97@gmail.com

  Linkedin : Krishna Salve

  GitHub : KrishnaSalve


- **Project Repository**

      https://github.com/KrishnaSalve/Laptop-price-prediction
