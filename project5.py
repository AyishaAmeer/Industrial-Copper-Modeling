from datetime import date
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

def streamlit_config():
    st.set_page_config(page_title='Industrial Copper Modeling',layout='wide')
    page_background_color="""<style>[data-testid="stHeader"]{background: rgba(0,0,0,0);}</style>"""
    st.markdown(page_background_color, unsafe_allow_html=True)
    st.title(":violet[INDUSTRIAL COPPER MODELING]")
    #st.markdown(f'<h1 style="text-align: center;">Industrial Copper Modeling</h1>',unsafe_allow_html=True)

def style_submit_button():
    st.markdown("""
                   <style>
                   div.stButton > button:first-child{
                   background-color: #367F89;
                   color: white;
                   width: 70%}
                   </style>
                """, unsafe_allow_html=True )

def style_prediction():
    st.markdown("""
                <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
            unsafe_allow_html=True
        )

class options:
    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
                    78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

    status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                    'Wonderful', 'Revised', 'Offered', 'Offerable']
    status_dict = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}

    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
    item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                        27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                        59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 
                        640405, 640665, 164141591, 164336407, 164337175, 929423819, 
                        1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
                        1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
                        1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 
                        1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

class prediction:
    def regression():
        with st.form('Regression'):
            col1,col2,col3 = st.columns([0.5,0.1,0.5])

            with col1:

                item_date = st.date_input(label='Item Date', min_value=date(2020,7,1), 
                                        max_value=date(2021,5,31), value=date(2020,7,1))
                
                quantity_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')

                country = st.selectbox(label='Country', options=options.country_values)

                item_type = st.selectbox(label='Item Type', options=options.item_type_values)

                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)

                product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)

            
            with col3:

                delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,8,1), 
                                            max_value=date(2022,2,28), value=date(2020,8,1))
                
                customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')

                status = st.selectbox(label='Status', options=options.status_values)

                application = st.selectbox(label='Application', options=options.application_values)

                width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)

                st.write('')
                st.write('')
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()

        col1,col2=st.columns([0.65,0.35])
        with col2:
            st.caption(body='*Min and Max values are reference only')
        if button:
            with open(r'C:\Users\ayish\GuviProjects\Industrial-copper Project\regression_model.pkl', 'rb') as f:
                model=pickle.load(f)
            
            user_data = np.array([[customer, 
                                country, 
                                options.status_dict[status], 
                                options.item_type_dict[item_type], 
                                application, 
                                width, 
                                product_ref, 
                                np.log(float(quantity_log)), 
                                np.log(float(thickness_log)),
                                item_date.day, item_date.month, item_date.year,
                                delivery_date.day, delivery_date.month, delivery_date.year]])
            y_pred=model.predict(user_data)
            selling_price=np.exp(y_pred[0])
            selling_price=round(selling_price,2)
            return selling_price
        
    def classification():
        with st.form('classification'):
            col1,col2,col3=st.columns([0.5,0.1,0.5])

            with col1:
                with col1:

                    item_date = st.date_input(label='Item Date', min_value=date(2020,7,1), 
                                            max_value=date(2021,5,31), value=date(2020,7,1))
                    
                    quantity_log = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: 1000000000)')

                    country = st.selectbox(label='Country', options=options.country_values)

                    item_type = st.selectbox(label='Item Type', options=options.item_type_values)

                    thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)

                    product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)


            with col3:

                delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,8,1), 
                                            max_value=date(2022,2,28), value=date(2020,8,1))
                
                customer = st.text_input(label='Customer ID (Min: 12458000 & Max: 2147484000)')

                selling_price_log = st.text_input(label='Selling Price (Min: 0.1 & Max: 100001000)')

                application = st.selectbox(label='Application', options=options.application_values)

                width = st.number_input(label='Width', min_value=1.0, max_value=2990000.0, value=1.0)

                st.write('')
                st.write('')
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()
        
        
        # give information to users
        col1,col2 = st.columns([0.65,0.35])
        with col2:
            st.caption(body='*Min and Max values are reference only')


        # user entered the all input values and click the button
        if button:
            
            # load the classification pickle model
            with open(r'C:\Users\ayish\GuviProjects\Industrial-copper Project\classification_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # make array for all user input values in required order for model prediction
            user_data = np.array([[customer, 
                                country, 
                                options.item_type_dict[item_type], 
                                application, 
                                width, 
                                product_ref, 
                                np.log(float(quantity_log)), 
                                np.log(float(thickness_log)),
                                np.log(float(selling_price_log)),
                                item_date.day, item_date.month, item_date.year,
                                delivery_date.day, delivery_date.month, delivery_date.year]])
            
            # model predict the status based on user input
            y_pred = model.predict(user_data)

            # we get the single output in list, so we access the output using index method
            status = y_pred[0]

            return status

streamlit_config()

with st.sidebar:
    select=option_menu("Main menu",["HOME","DATA INSIGHTS","DATA PREDICTION"],icons=["house","upload","gear"])

if select=="HOME":
    
    # Set the background image
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://techgenix.com/tgwordpress/wp-content/uploads/2019/05/smart-factory-shutterstock_567310369-1024x522.jpg");
        
        
    }
    </style>
    """
    #st.markdown(background_image, unsafe_allow_html=True)
    img=Image.open(r'C:\Users\ayish\GuviProjects\Industrial-copper Project\smart_industry.png')
    st.image(img,width=1000,channels="RGB")
    st.markdown("## :green[ **Introduction:**] ")
    st.markdown('## :rainbow[An industrial copper modeling problem typically involves the development of mathematical, statistical, or machine learning models to predict or optimize various aspects related to the production, utilization, or market behavior of copper in an industrial context.]')
    st.markdown("## :green[ **Skills take away From This Project:**] ")
    st.markdown('## :rainbow[Python scripting, Data Preprocessing, EDA, Streamlit]')
    st.markdown("## :green[ **Project Approach:**] ")
    st.markdown("## :rainbow[:point_right:**Data Understanding:**] Identifying the types of variables (continuous or categorical) ")
    st.markdown("## :rainbow[:point_right:**Data Preprocessing:**] Handle missing values, Treat Outliers using IQR, Identify Skewness and Encode categorical variables using suitable techniques. ")
    st.markdown("## :rainbow[:point_right:**EDA:**] Visualizing outliers and skewness using Seaborn's boxplot, distplot, violinplot.")
    st.markdown("## :rainbow[:point_right:**Feature Engineering:**] Drop highly correlated columns using SNS HEATMAP.")
    st.markdown("## :rainbow[:point_right:**Model Building and Evaluation:**] Split the dataset into training and testing/validation sets. Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.Interpret the model results and assess its performance. Same steps for Regression modelling.")
    st.markdown("## :rainbow[:point_right:**Model GUI:**] Using streamlit module create interactive page. ")
    st.markdown("## :rainbow[:point_right:**Pickle:**] Using pickle module to dump and load models. ")
elif select=="DATA INSIGHTS":
    
    st.markdown('## :green[ **The learning outcomes of this project are:**]')
    st.markdown('## :orange[:point_right:**Developing proficiency in Python programming language and its data analysis libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.**]')
    st.markdown('## :orange[:point_right:**Gaining experience in data preprocessing techniques such as handling missing values, outlier detection, and data normalization to prepare data for machine learning modeling.**]')
    st.markdown('## :orange[:point_right:**Understanding and visualizing the data using EDA techniques such as boxplots, histograms, and scatter plots.**]')
    st.markdown('## :orange[:point_right:**Learning and applying advanced machine learning techniques such as regression and classification to predict continuous and binary target variables, respectively.**]')
    st.markdown('## :orange[:point_right:**Building and optimizing machine learning models using appropriate evaluation metrics and techniques such as cross-validation and grid search.**]')
    st.markdown('## :orange[:point_right:**Experience in feature engineering techniques to create new informative representations of the data.**]')
    st.markdown('## :orange[:point_right:**Developing a web application using the Streamlit module to showcase the machine learning models and make predictions on new data.**]')
    st.markdown('## :orange[:point_right:**Understanding the challenges and best practices in the manufacturing domain and how machine learning can help solve them.**]')
    st.markdown('## :green[ **About the dataset attributes:**]')
    st.markdown("## :orange[**1. 'id':**][ This column likely serves as a unique identifier for each transaction or item, which can be useful for tracking and record-keeping.]")
    st.markdown("## :orange[**2. 'item date':**][ This column represents the date when each transaction or item was recorded or occurred. It's important for tracking the timing of business activities.]")
    st.markdown("## :orange[**3. 'quantity tons':**][ This column indicates the quantity of the item in tons, which is essential for inventory management and understanding the volume of products sold or produced.]")
    st.markdown("## :orange[**4. 'customer':**][ The 'customer' column refers to the name or identifier of the customer who either purchased or ordered the items. It's crucial for maintaining customer relationships and tracking sales.]")
    st.markdown("## :orange[**5. 'country':**][ The 'country' column specifies the country associated with each customer. This information can be useful for understanding the geographic distribution of customers and may have implications for logistics and international sales.]")
    st.markdown("## :orange[**6. 'status':**][ The 'status' column likely describes the current status of the transaction or item. This information can be used to track the progress of orders or transactions, such as 'Draft' or 'Won'.]")
    st.markdown("## :orange[**7. 'item type':**][ This column categorizes the type or category of the items being sold or produced. Understanding item types is essential for inventory categorization and business reporting.]")
    st.markdown("## :orange[**8. 'application':**][ The 'application' column defines the specific use or application of the items. This information can help tailor marketing and product development efforts.]")
    st.markdown("## :orange[**9. 'thickness':**][ The 'thickness' column provides details about the thickness of the items. It's critical when dealing with materials where thickness is a significant factor, such as metals or construction materials.]")
    st.markdown("## :orange[**10. 'width':**][ The 'width' column specifies the width of the items. It's important for understanding the size and dimensions of the products.]")
    st.markdown("## :orange[**11. 'material_ref':**][ This column appears to be a reference or identifier for the material used in the items. It's essential for tracking the source or composition of the products.]")
    st.markdown("## :orange[**12. 'product_ref':**][ The 'product_ref' column seems to be a reference or identifier for the specific product. This information is useful for identifying and cataloging products in a standardized way.]")
    st.markdown("## :orange[**13. 'delivery date':**][ This column records the expected or actual delivery date for each item or transaction. It's crucial for managing logistics and ensuring timely delivery to customers.]")
    st.markdown("## :orange[**14. 'selling_price':**][ The 'selling_price' column represents the price at which the items are sold. This is a critical factor for revenue generation and profitability analysis.]")
elif select=="DATA PREDICTION":
    tab1, tab2=st.tabs(['PREDICT SELLING PRICE','PREDICT STATUS'])
    with tab1:
        try:
            selling_price = prediction.regression()

            if selling_price:
                # apply custom css style for prediction text
                style_prediction()
                st.markdown(f'### <div class="center-text">Predicted Selling Price = {selling_price}</div>', unsafe_allow_html=True)
        except ValueError:

            col1,col2,col3 = st.columns([0.26,0.55,0.26])

            with col2:
                st.warning('##### Quantity Tons / Customer ID is empty')
    with tab2:
        try:
            status=prediction.classification()
            if status == 1:            
                style_prediction()
                st.markdown(f'### <div class="center-text">Predicted Status = Won</div>', unsafe_allow_html=True)
                
                
            elif status == 0:
                style_prediction()
                st.markdown(f'### <div class="center-text">Predicted Status = Lost</div>', unsafe_allow_html=True)
                
        except ValueError:

            col1,col2,col3 = st.columns([0.15,0.70,0.15])

            with col2:
                st.warning('##### Quantity Tons / Customer ID / Selling Price is empty')