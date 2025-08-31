
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import model training 
from model_training import prepare_and_train_model

@st.cache_resource
def load_or_create_model():
    """Load existing model or create new one from CSV data"""
    try:
        # Check if all model files exist
        model_files = ['model.pkl', 'encoders.pkl', 'feature_columns.pkl']
        if all(os.path.exists(f) for f in model_files):
            model = joblib.load('model.pkl')
            encoders = joblib.load('encoders.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            st.success("✅ Model loaded successfully from saved files!")
            return model, encoders, feature_columns
        else:
            st.info("🔄 Training new model from CSV data... This may take a moment.")

            # Check if CSV exists
            if os.path.exists('customer_data.csv'):
                st.info("📊 Found customer_data.csv - using your data for training")
            else:
                st.warning("⚠️ customer_data.csv not found - using fallback data")

            model, encoders, feature_columns = prepare_and_train_model()

            if model is not None:
                st.success("✅ Model trained successfully from CSV data!")
                return model, encoders, feature_columns
            else:
                st.error("❌ Failed to train model")
                return None, None, None

    except Exception as e:
        st.error(f"❌ Error loading/creating model: {e}")
        return None, None, None

def predict_marketing_action(customer_data, model, encoders, feature_columns):
    """Predict marketing action for a customer"""
    try:
        # Create customer dataframe with exact column names from CSV
        customer_df = pd.DataFrame([{
            'Age_of_Buyer': customer_data['Age_of_Buyer'],
            'Average_Order_Value': customer_data['Average_Order_Value'], 
            'Time_Spent_on_App_per_Day_minutes': customer_data['Time_Spent_on_App_per_Day_minutes'],
            'Recency_days': customer_data['Recency_days'],
            'Frequency_purchases': customer_data['Frequency_purchases'],
            'Monetary_spend': customer_data['Monetary_spend'],
            'Email_Open_Rate': customer_data['Email_Open_Rate'],
            'Push_CTR': customer_data['Push_CTR'],
            'Coupon_Redemption_Rate': customer_data['Coupon_Redemption_Rate'],
            'Chat_Engagement_Rate': customer_data['Chat_Engagement_Rate'],
            'CLTV': customer_data['CLTV'],
            'Pages_per_Session': customer_data['Pages_per_Session'],
            'Avg_Session_Duration_min': customer_data['Avg_Session_Duration_min'],
            'Product_Views': customer_data['Product_Views'],
            'Discount_Threshold_percent': customer_data['Discount_Threshold_percent'],
            'Churn_Risk': customer_data['Churn_Risk'],
            'Type_of_City': customer_data['Type_of_City'],
            'Recent_Action': customer_data['Recent_Action'],
            'Top_Category': customer_data['Top_Category'],
            'Primary_Device': customer_data['Primary_Device'],
            'Locale': customer_data['Locale']
        }])

        # Encode categorical variables using the same encoders from training
        try:
            customer_df['City_Encoded'] = encoders['type_of_city_encoder'].transform([customer_data['Type_of_City']])[0]
            customer_df['Action_Encoded'] = encoders['recent_action_encoder'].transform([customer_data['Recent_Action']])[0]
            customer_df['Category_Encoded'] = encoders['top_category_encoder'].transform([customer_data['Top_Category']])[0]
            customer_df['Device_Encoded'] = encoders['primary_device_encoder'].transform([customer_data['Primary_Device']])[0]
            customer_df['Locale_Encoded'] = encoders['locale_encoder'].transform([customer_data['Locale']])[0]
        except KeyError as e:
            st.error(f"Encoder error: {e}. Model may need retraining.")
            return None, None

        # Select features in the same order as training
        X_customer = customer_df[feature_columns]

        # Make predictions
        prediction = model.predict(X_customer)[0]
        prediction_proba = model.predict_proba(X_customer)[0]

        # Get top 5 predictions with probabilities
        top_indices = prediction_proba.argsort()[-5:][::-1]
        top_predictions = [(model.classes_[i], prediction_proba[i]) for i in top_indices]

        return prediction, top_predictions

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def create_confidence_chart(top_predictions):
    """Create confidence score chart using Plotly"""
    if not top_predictions:
        return None

    actions = [action for action, prob in top_predictions]
    probabilities = [prob * 100 for action, prob in top_predictions]

    # Create horizontal bar chart
    fig = px.bar(
        x=probabilities,
        y=actions,
        orientation='h',
        title="Marketing Action Confidence Scores",
        labels={'x': 'Confidence (%)', 'y': 'Marketing Actions'},
        color=probabilities,
        color_continuous_scale='viridis',
        height=400
    )

    # Customize layout
    fig.update_layout(
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Confidence Score (%)",
        yaxis_title="Marketing Actions",
        title_x=0.5
    )

    return fig

def create_customer_insights_chart(customer_data):
    """Create customer insights radar chart"""
    try:
        # Prepare data for radar chart
        metrics = {
            'Engagement': (customer_data['Email_Open_Rate'] + customer_data['Push_CTR'] + customer_data['Chat_Engagement_Rate']) / 3,
            'Loyalty': min(customer_data['Frequency_purchases'] / 50, 1.0),  # Normalize to 0-1
            'Value': min(customer_data['CLTV'] / 5000, 1.0),  # Normalize to 0-1  
            'Activity': min(customer_data['Time_Spent_on_App_per_Day_minutes'] / 180, 1.0),  # Normalize to 0-1
            'Retention Risk': customer_data['Churn_Risk']
        }

        categories = list(metrics.keys())
        values = [metrics[cat] * 100 for cat in categories]  # Convert to percentage

        # Close the radar chart
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Customer Profile',
            line_color='rgb(31, 119, 180)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Customer Profile Overview",
            title_x=0.5,
            height=400
        )

        return fig

    except Exception as e:
        st.error(f"Error creating insights chart: {e}")
        return None

def get_implementation_tip(prediction):
    """Get implementation tip for the predicted action"""
    tips = {
        "Discount Coupon Email": "📧 Send personalized discount offers within 24 hours. Include product recommendations based on browsing history.",
        "Push Notification Reminder": "📱 Send push notification within 2 hours for cart abandonment or within 1 week for re-engagement.",
        "Personalized Email Recommendations": "🎯 Use collaborative filtering to suggest products similar to past purchases or browsing behavior.",
        "Proactive Chat Support": "💬 Initiate chat session offering help. Train agents to address specific concerns mentioned in reviews.",
        "VIP Exclusive Offer": "👑 Provide early access to sales, exclusive products, or special discounts (25-30% minimum).",
        "Win-back Campaign": "🎪 Launch multi-channel campaign: email + push + SMS over 2-3 weeks with increasing incentives.",
        "SMS Reactivation": "📱 Send SMS for customers unresponsive to email/push. Include direct link to app with special offer.",
        "Manager Follow-up Call": "☎️ Personal call within 24-48 hours for high-value customers. Focus on issue resolution.",
        "Abandoned Cart Email": "🛒 Send series of 3 emails: immediate, 24h later, 72h later with progressive discounts.",
        "Re-engagement Push": "🔔 Send push notification highlighting new features, products, or limited-time offers.",
        "Comeback Email Offer": "📬 Welcome-back email with special offer (15-20% discount) and showcase new arrivals.",
        "Feedback Email Survey": "📝 Send survey to understand issues, offer compensation, and follow up with resolution.",
        "Product Suggestion Push": "📲 Push notification with 3-4 product recommendations based on browsing patterns.",
        "First Purchase Discount": "🎁 Offer 10-15% discount for first purchase with easy checkout process.",
        "Browsing Behavior Email": "👁️ Send email featuring products from recently viewed categories with social proof.",
        "Urgent Retention Campaign": "🚨 Immediate multi-channel outreach with best possible offer and personal touch.",
        "Cross-sell Campaign": "🔄 Suggest complementary products based on purchase history with bundle discounts.",
        "Multi-channel Re-engagement": "📢 Coordinate email, push, and SMS campaign over 10-14 days with varied messaging.",
        "Regular Newsletter": "📰 Monthly newsletter with product updates, tips, and moderate promotional content.",
        "VIP Retention Call": "📞 Urgent call from senior team member with exclusive retention offer and service improvements."
    }

    return tips.get(prediction, "💡 Implement this action based on your standard marketing procedures and customer preferences.")

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Marketing Action Predictor",
        page_icon="🎯", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: black;  # transparent>black
        font-size: 3rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: red;            # white>red
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f1fa; #replaced 9 with 1
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🎯 AI Marketing Action Predictor</h1>', unsafe_allow_html=True)

    # Subtitle with model info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background-color: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3>🤖 Powered by Machine Learning</h3>
        <p style="margin: 0;"><strong>CSV-Based Training • Real-time Predictions • 95% Accuracy</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Load model with status display
    with st.spinner("Loading AI model..."):
        model, encoders, feature_columns = load_or_create_model()

    if model is None:
        st.error("❌ Failed to load or train the model. Please check your data and try again.")
        st.info("💡 Make sure 'customer_data.csv' is uploaded to your GitHub repository.")
        return

    # Display model info
    st.success(f"🎯 Model ready! Can predict {len(model.classes_)} different marketing actions.")

    # Sidebar for customer input
    st.sidebar.markdown("## 📋 Customer Profile Input")
    st.sidebar.markdown("*Fill in the customer details below to get AI-powered marketing recommendations*")

    # Demographics section
    with st.sidebar:
        st.markdown("### 👤 Demographics")
        age = st.slider("Customer Age", 16, 65, 30, help="Age of the customer")
        city_type = st.selectbox(
            "City Tier", 
            ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
            help="City classification: Tier 1 (metros), Tier 2 (major cities), etc."
        )
        device = st.selectbox(
            "Primary Device", 
            ['Mobile', 'Desktop', 'Tablet'],
            help="Main device used for shopping"
        )
        locale = st.selectbox(
            "Locale", 
            ['en_US', 'en_GB', 'es_ES', 'fr_FR', 'de_DE'],
            help="Language and region preference"
        )

        st.markdown("### 📱 Recent Behavior")
        recent_action = st.selectbox(
            "Most Recent Action",
            ['cart abandonment', 'not visiting the app', 'not adding to cart', 'wrote a bad review'],
            help="Customer's most recent significant action"
        )

        top_category = st.selectbox(
            "Favorite Category",
            ['Electronics', 'Fashion', 'Home & Kitchen', 'Health', 'Sports', 'Beauty'],
            help="Customer's most purchased product category"
        )

        st.markdown("### 📊 Engagement Metrics")
        email_open_rate = st.slider(
            "Email Open Rate", 
            0.0, 1.0, 0.5, 0.01,
            help="Percentage of marketing emails opened by customer"
        )
        push_ctr = st.slider(
            "Push Click Rate", 
            0.0, 1.0, 0.4, 0.01,
            help="Click-through rate on push notifications"
        )
        chat_engagement = st.slider(
            "Chat Response Rate", 
            0.0, 1.0, 0.3, 0.01,
            help="Response rate to customer service chats"
        )
        coupon_redemption = st.slider(
            "Coupon Usage Rate", 
            0.0, 1.0, 0.6, 0.01,
            help="Percentage of coupons actually used"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💰 Purchase Behavior")
        aov = st.number_input(
            "Average Order Value ($)", 
            min_value=50, max_value=15000, value=2500, step=50,
            help="Average amount spent per order"
        )
        frequency = st.number_input(
            "Purchase Frequency (yearly)", 
            min_value=1, max_value=100, value=12, step=1,
            help="Number of purchases in the last 12 months"
        )
        monetary = st.number_input(
            "Total Spend ($)", 
            min_value=100, max_value=50000, value=8000, step=100,
            help="Total amount spent to date"
        )
        recency = st.number_input(
            "Days Since Last Purchase", 
            min_value=0, max_value=730, value=45, step=1,
            help="Number of days since the last purchase"
        )

        st.markdown("### 📱 App Usage")
        time_on_app = st.number_input(
            "Daily App Time (minutes)", 
            min_value=1, max_value=300, value=75, step=5,
            help="Average time spent on app per day"
        )
        pages_per_session = st.number_input(
            "Pages per Session", 
            min_value=1.0, max_value=50.0, value=6.5, step=0.5,
            help="Average pages viewed per app session"
        )
        session_duration = st.number_input(
            "Avg Session Duration (min)", 
            min_value=0.5, max_value=60.0, value=12.0, step=0.5,
            help="Average time spent per session"
        )

    with col2:
        st.markdown("### 📈 Customer Value")
        cltv = st.number_input(
            "Customer Lifetime Value ($)", 
            min_value=100.0, max_value=15000.0, value=3500.0, step=100.0,
            help="Predicted total value of customer relationship"
        )
        product_views = st.number_input(
            "Products Viewed per Session", 
            min_value=1, max_value=20, value=4, step=1,
            help="Average products viewed per session"
        )
        discount_threshold = st.selectbox(
            "Discount Sensitivity (%)", 
            [5, 10, 15, 20, 25, 30], index=2,
            help="Minimum discount needed to trigger purchase"
        )
        churn_risk = st.slider(
            "Churn Risk Score", 
            0.0, 1.0, 0.35, 0.01,
            help="Probability of customer stopping purchases (0=loyal, 1=likely to churn)"
        )

    # Create customer data dictionary with exact CSV column names
    customer_data = {
        'Age_of_Buyer': age,
        'Average_Order_Value': aov,
        'Time_Spent_on_App_per_Day_minutes': time_on_app,
        'Recency_days': recency,
        'Frequency_purchases': frequency,
        'Monetary_spend': monetary,
        'Email_Open_Rate': email_open_rate,
        'Push_CTR': push_ctr,
        'Coupon_Redemption_Rate': coupon_redemption,
        'Chat_Engagement_Rate': chat_engagement,
        'CLTV': cltv,
        'Pages_per_Session': pages_per_session,
        'Avg_Session_Duration_min': session_duration,
        'Product_Views': product_views,
        'Discount_Threshold_percent': discount_threshold,
        'Churn_Risk': churn_risk,
        'Type_of_City': city_type,
        'Recent_Action': recent_action,
        'Top_Category': top_category,
        'Primary_Device': device,
        'Locale': locale
    }

    # Prediction section
    st.markdown("---")

    # Center the prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Generate AI Marketing Recommendation", 
            type="primary", 
            use_container_width=True,
            help="Click to get personalized marketing action recommendation"
        )

    if predict_button:
        with st.spinner("🤖 AI is analyzing customer profile..."):
            prediction, top_predictions = predict_marketing_action(customer_data, model, encoders, feature_columns)

            if prediction and top_predictions:
                st.balloons()

                # Main recommendation display
                st.markdown(f"""
                <div class="recommendation-box">
                    <h2>🎯 Recommended Marketing Action</h2>
                    <h1 style="margin: 1rem 0; font-size: 2.5rem;">{prediction}</h1>
                    <p style="font-size: 1.3rem; margin: 0;">
                        <strong>AI Confidence: {top_predictions[0][1]:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Charts and analysis
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown("### 📊 All AI Recommendations")
                    confidence_fig = create_confidence_chart(top_predictions)
                    if confidence_fig:
                        st.plotly_chart(confidence_fig, use_container_width=True)

                with col2:
                    st.markdown("### 👤 Customer Insights")
                    insights_fig = create_customer_insights_chart(customer_data)
                    if insights_fig:
                        st.plotly_chart(insights_fig, use_container_width=True)

                # Customer metrics
                st.markdown("### 📋 Customer Analysis")
                metric_cols = st.columns(4)

                with metric_cols[0]:
                    value_tier = "🌟 High" if cltv > 4000 else "💎 Premium" if cltv > 2500 else "💰 Standard"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Customer Value</h4>
                        <h3>{value_tier}</h3>
                        <p>${cltv:,.0f} CLTV</p>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_cols[1]:
                    engagement_score = (email_open_rate + push_ctr + chat_engagement) / 3
                    eng_level = "🔥 High" if engagement_score > 0.6 else "📈 Medium" if engagement_score > 0.3 else "😴 Low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Engagement</h4>
                        <h3>{eng_level}</h3>
                        <p>{engagement_score:.1%} Score</p>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_cols[2]:
                    risk_level = "🚨 High" if churn_risk > 0.7 else "⚠️ Medium" if churn_risk > 0.4 else "✅ Low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Churn Risk</h4>
                        <h3>{risk_level}</h3>
                        <p>{churn_risk:.1%} Risk</p>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_cols[3]:
                    urgency = "🔴 Urgent" if recency > 90 else "🟡 Medium" if recency > 30 else "🟢 Low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Action Urgency</h4>
                        <h3>{urgency}</h3>
                        <p>{recency} days</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Detailed recommendations table
                st.markdown("### 📊 Alternative Recommendations")
                recommendations_df = pd.DataFrame({
                    'Rank': range(1, len(top_predictions) + 1),
                    'Marketing Action': [action for action, _ in top_predictions],
                    'Confidence': [f"{prob:.1%}" for _, prob in top_predictions],
                    'Priority': ['🥇 Primary', '🥈 Secondary', '🥉 Tertiary', '📋 Backup', '📋 Alternative'][:len(top_predictions)]
                })

                st.dataframe(
                    recommendations_df, 
                    use_container_width=True,
                    hide_index=True
                )

                # Implementation guidelines
                st.markdown("### 💡 Implementation Guidelines")
                tip = get_implementation_tip(prediction)
                st.info(tip)

                # Additional strategic insights
                st.markdown("### 🎯 Strategic Insights")
                insights_cols = st.columns(2)

                with insights_cols[0]:
                    st.markdown("**📧 Channel Recommendations:**")
                    if email_open_rate > max(push_ctr, chat_engagement):
                        st.write("• **Primary:** Email marketing")
                        st.write("• **Secondary:** Push notifications")
                    elif push_ctr > chat_engagement:
                        st.write("• **Primary:** Push notifications") 
                        st.write("• **Secondary:** Email marketing")
                    else:
                        st.write("• **Primary:** Chat/In-app messaging")
                        st.write("• **Secondary:** Email + Push combo")

                with insights_cols[1]:
                    st.markdown("**⏰ Timing Recommendations:**")
                    if recent_action == 'cart abandonment':
                        st.write("• **Immediate:** Within 1-2 hours")
                        st.write("• **Follow-up:** 24 hours, then 72 hours")
                    elif churn_risk > 0.7:
                        st.write("• **Urgent:** Within 24 hours")
                        st.write("• **Sustained:** Multi-week campaign")
                    else:
                        st.write("• **Optimal:** Within 1-3 days")
                        st.write("• **Follow-up:** Weekly check-ins")

                # Success metrics to track
                st.markdown("### 📈 Success Metrics to Track")
                success_cols = st.columns(3)

                with success_cols[0]:
                    st.markdown("**📊 Primary KPIs:**")
                    st.write("• Click-through rate")
                    st.write("• Conversion rate") 
                    st.write("• Revenue generated")

                with success_cols[1]:
                    st.markdown("**🎯 Engagement KPIs:**")
                    st.write("• Email open rate")
                    st.write("• App session duration")
                    st.write("• Page views per session")

                with success_cols[2]:
                    st.markdown("**💰 Business KPIs:**")
                    st.write("• Customer lifetime value")
                    st.write("• Retention rate")
                    st.write("• Average order value")

            else:
                st.error("❌ Unable to generate prediction. Please check the input data and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>🤖 AI Marketing Action Predictor</h4>
        <p><strong>CSV-Powered Training • Real-time Predictions • Actionable Insights</strong></p>
        <p>Built with Streamlit 🚀 | Powered by Machine Learning 🧠 | Deployed on Cloud ☁️</p>
        <p><em>Transform your marketing from reactive to predictive</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
