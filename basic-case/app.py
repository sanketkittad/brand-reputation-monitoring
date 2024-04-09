from flask import Flask,render_template
from flask import request
app=Flask(__name__)
import mpld3
import re 
import io 
import base64
import numpy as np


from textblob import TextBlob 

import matplotlib.pyplot as plt

import pandas as pd

from wordcloud import WordCloud

from better_profanity import profanity

import yake

import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx


    
features_array={
        "mobiles":{
        "camera": ["camera", "photo", "picture", "lens", "image quality"], 
        "battery": ["battery", "charge", "life", "battery drain"],
        "screen": ["screen", "display", "resolution", "oled", "amoled"],
        "processor": ["processor", "speed", "chip", "snapdragon", "performance", "bionic","a14"], 
        "design": ["design", "look", "feel", "build quality"],
        "price": ["price", "expensive", "cheap", "value"]
    },
    
    "food_delivery": {
        "ordering": ["order", "ordering", "menu", "easy", "variety", "options"],
        "delivery": ["delivery", "time", "fast", "track", "tracking", "fee", "fees"],
        "food": ["food", "quality", "taste", "delicious", "portion"],
        "customer_service": ["service", "support", "help", "response", "refund"],
        "pricing": ["price", "value", "expensive", "cheap", "discount", "promotion"]
    },
    "laptops": {
        "performance": ["performance", "speed", "processor", "i5", "i7", "i9", "ryzen", "m1", "m2"],
        "graphics": ["graphics", "gpu", "dedicated_graphics", "integrated_graphics", "nvidia", "amd", "intel_iris"],
        "display": ["display", "screen", "resolution", "retina", "oled", "ips", "refresh_rate"],
        "storage": ["storage", "ssd", "hdd", "hard_drive", "capacity", "256gb", "512gb", "1tb"],
        "memory": ["memory", "ram", "8gb", "16gb", "32gb"],
        "build": ["build", "design", "materials", "aluminum", "magnesium", "weight", "thinness"],
        "battery": ["battery", "battery_life", "charge", "longevity"],
        "price": ["price", "expensive", "cheap", "value", "affordable"]
    },
    "streaming_services": {
        "content_library": ["library", "movies", "shows", "selection", "variety", "originals", "exclusives"],
        "genres": ["genres", "comedy", "drama", "thriller", "sci-fi", "horror", "documentaries"],
        "quality": ["quality", "4k", "hdr", "dolby", "resolution", "streaming_quality", "buffering"],
        "price": ["price", "value", "subscription", "plans", "bundles", "free_trial"],
        "customer_support": ["support", "customer_service", "help", "responsive"]
    },
    "social_media": {
        "content_types": ["posts", "photos", "videos", "stories", "reels", "live"],
        "user_experience": ["ui", "ux", "user_interface", "design", "navigation", "ease_of_use", "bugs"],
        "social_features": ["messaging", "groups", "events", "marketplace", "sharing", "profiles"],
        "community": ["community", "interactions", "comments", "engagement", "moderation", "toxicity"],
        "privacy": ["privacy", "security", "data", "data_collection", "ads", "settings"],
        "customer_support": ["support", "customer_service", "help", "responsive", "issues"]
    },
    "automobiles": {
        "performance": ["performance", "speed", "acceleration", "handling", "horsepower", "torque", "engine", "transmission"],
        "fuel_efficiency": ["fuel", "mpg", "efficiency", "hybrid", "electric", "range"],
        "safety": ["safety", "crash_test", "airbags", "brakes", "driver_assistance"],
        "reliability": ["reliability", "dependable", "issues", "maintenance", "warranty"],
        "technology": ["technology", "infotainment", "gps", "bluetooth", "sound_system", "voice_commands"],
        "price": ["price", "value", "affordability", "expensive", "deals"]
    },
    "music_streaming_services": {
        "music_library": ["library", "songs", "artists", "albums", "selection", "variety"],
        "genres": ["genres", "pop", "rock", "hip_hop", "classical", "jazz", "electronic", "niche"],
        "sound_quality": ["quality", "high_fidelity", "lossless", "bitrate", "spatial_audio"],
        "features": ["offline_listening", "lyrics", "podcasts", "social", "concerts"],
        "price": ["price", "value", "subscription", "plans", "bundle", "free"],
        "customer_support": ["support", "customer_service", "help", "responsive"]
    },
    "gaming_consoles": {
        "hardware": ["hardware", "power", "graphics", "specs", "processor", "storage", "controller"],
        "online_services": ["online", "multiplayer", "subscriptions", "ps_plus", "game_pass"],
        "features": ["backward_compatibility", "vr", "game_streaming", "interface", "media"],
        "game_library": ["library", "games", "selection", "variety", "indie"],
        "user_experience": ["ui", "ux", "user_interface", "design", "navigation", "ease_of_use"],
        "community": ["community", "online", "friends", "tournaments"],
        "price": ["price", "value", "affordability", "expensive"]
    }
}




@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        brand1=request.form['brand1']
        brand2=request.form['brand2']
        category=request.form['category']
        jsonobjs=getAnalysis(brand1,brand2,category)
        return render_template("main.html",jsonobjs=jsonobjs)    
    return render_template("index.html")
    
# Initialize Sentiment analyzer
def getAnalysis(brand1,brand2,category):
    analyzer = SentimentIntensityAnalyzer() 
    brand1=brand1.lower()
    brand2=brand2.lower()
    features=features_array[category]
    buff=io.BytesIO()
    def calculate_sentiment(text):
        if not text.strip():  # Check if empty after removing whitespace
            return 0  # Return 0 for empty tweets 
        score = analyzer.polarity_scores(text)['compound']
        return score

    def process_data(brand1_path, brand2_path):
        results = []
        for filename in os.listdir(brand1_path) + os.listdir(brand2_path): 
            if filename.startswith(brand1[0]+'_'):
                brand = brand1
            elif filename.startswith(brand2[0]+'_'):
                brand = brand2
            else:
                continue  

            # Extract date from filename (dd-mm-yyyy format)
            date_str = filename.split('_')[2] + "-" + filename.split('_')[1] + "-" + filename.split('_')[3].split('.')[0] 

            # Reformat date for sorting  (mm-dd-yyyy)
            mmddyyyy_date_str = date_str[3:5] + "-" + date_str[:2] + "-" + date_str[6:]

            filepath = os.path.join(brand1_path if brand == brand1 else brand2_path, filename)
            df = pd.read_csv(filepath)

            # Calculate daily sentiment 
            daily_sentiment = df['tweet_text_element'].apply(calculate_sentiment).mean()

            # Find or create the matching date entry 
            existing_entry = next((item for item in results if item['day'] == mmddyyyy_date_str), None) 
            if existing_entry:
                existing_entry['brand1_sentiment'] = daily_sentiment
            else:   
                results.append({
                    'day': mmddyyyy_date_str,  
                    'brand2_sentiment': daily_sentiment 
                })

        # Sort after processing all files 
        results.sort(key=lambda item: item['day'])  

        return pd.DataFrame(results)

    brand1_path = brand1[0].upper()+brand1[1:]
    brand2_path = brand2[0].upper()+brand2[1:]
    dfr = process_data(brand1_path, brand2_path)
    # index=False to avoid an extra index column
    print(dfr)



    def calculate_influence(df):
        # Normalize sentiment scores
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        df[['brand1_sentiment', 'brand2_sentiment']] = scaler.fit_transform(df[['brand1_sentiment', 'brand2_sentiment']])

        # Correlation-based influence
        correlation = np.corrcoef(df['brand1_sentiment'], df['brand2_sentiment'])[0, 1]
        abs_correlation = abs(correlation)

        influence_brand1_on_brand2 = abs_correlation * 100 
        influence_brand2_on_brand1 = abs_correlation * 100 

        return influence_brand1_on_brand2, influence_brand2_on_brand1

    # Load the data


    # Calculate influences
    dtf=dfr.copy()
    influence_brand1_on_brand2, influence_brand2_on_brand1 = calculate_influence(dtf)

    print(f"Influence of {brand1} on {brand2}: {influence_brand1_on_brand2:.2f}%")
    print(f"Influence of {brand2} on {brand1}: {influence_brand2_on_brand1:.2f}%")


    brand = f"{brand1} on {brand2}"  
    influence_percentage = influence_brand1_on_brand2

    # Create data for the bar graph
    data = {'Influence Score (%)': [influence_percentage, 100 - influence_percentage]}
    df_d = pd.DataFrame(data, index=[brand, 'Remaining'])

    # Constants for styling
    SELECTED_COLOR = 'indianred'  

 
    # Create the bar graph
    bar1=plt.figure()
    
    bars = plt.bar(df_d.index, df_d['Influence Score (%)'], color='skyblue')
    plt.xlabel('Influence')
    plt.ylabel('Percentage (%)')
    plt.title(f"{brand1} Influence on Sentiment")
    plt.ylim(0, 100) 

    # Add percentage labels (optional)
    for bar in bars:
        value = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_y() + value  
        plt.text(x, y, f"{value:.1f}%", ha='center', va='bottom') 
    plt.tight_layout()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plot_url = base64.b64encode(buff.getvalue()).decode('utf8')
    jsonobjs={}

    jsonobjs["bar1"]=plot_url
    def process_data(brand1_path, brand2_path):
        all_data = []
        for path, brand in [(brand1_path, brand1[0].upper()+brand1[1:]), (brand2_path, brand2[0].upper()+brand2[1:])]:
            for filename in os.listdir(path):
                if filename.startswith(brand.lower()[0] + '_'):  
                    filepath = os.path.join(path, filename)
                    df = pd.read_csv(filepath)
                    df['brand'] = brand  
                    all_data.append(df)

        combined_df = pd.concat(all_data)
        return combined_df

    df = process_data(brand1_path, brand2_path)
    print(df.head)
    
    def generate_wordcloud(brand_data, brand_name,*ct):
        text = " ".join(tweet for tweet in brand_data['tweet_text_element'])
        wordcloud = WordCloud(background_color='white', width=800, height=600).generate(text)

        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for {brand_name}")
        buff=io.BytesIO()
        plt.savefig(buff,format='png')
        buff.seek(0)
        plot_url = base64.b64encode(buff.getvalue()).decode('utf8')

        jsonobjs[ct]=plot_url
        print(ct)



    # Generate word clouds for each brand

    # for brand in df['brand'].unique():
    #     brand_data = df[df['brand'] == brand]
    #     ct='bar2'
    #     generate_wordcloud(brand_data, brand,ct)
    #     ct='bar3'

    brand_data = df[df['brand'] ==     brand1[0].upper()+brand1[1:]]
    text = " ".join(tweet for tweet in brand_data['tweet_text_element'])
    wordcloud = WordCloud(background_color='white', width=800, height=600).generate(text)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {brand1[0].upper()+brand1[1:]}")
    buff=io.BytesIO()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plot_url = base64.b64encode(buff.getvalue()).decode('utf8')
    jsonobjs["bar2"]=plot_url
    print("bar2")    
    brand_data = df[df['brand'] ==     brand2[0].upper()+brand2[1:]]
    text = " ".join(tweet for tweet in brand_data['tweet_text_element'])
    wordcloud = WordCloud(background_color='white', width=800, height=600).generate(text)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {brand2[0].upper()+brand2[1:]}")
    buff=io.BytesIO()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plot_url = base64.b64encode(buff.getvalue()).decode('utf8')
    jsonobjs["bar3"]=plot_url
    print("bar3")
    sentiment_scores = {}  # Initialize an empty dictionary to store sentiment scores

    def calculate_feature_sentiment(tweet_text, feature_keywords):
        sentiment_score = 0  # Initialize
        if any(word in tweet_text for word in feature_keywords):
            sentiment_score = analyzer.polarity_scores(tweet_text)['compound']
        return sentiment_score


    for brand in df['brand'].unique():
        for feature, keywords in features.items():
            feature_df = df[df['brand'] == brand].copy()  # Create a copy here
            feature_df[f"{feature}_sentiment"] = feature_df['tweet_text_element'].apply(calculate_feature_sentiment, args=(keywords,))

            avg_sentiment = feature_df[f"{feature}_sentiment"].fillna(0).mean()
            # print(f"{brand} - {feature} Average Sentiment: {avg_sentiment:.3f}")
            
            # Store the calculated sentiment score in the dictionary
            if brand not in sentiment_scores:
                sentiment_scores[brand] = {}
            sentiment_scores[brand][feature] = avg_sentiment

    # Print the sentiment scores dictionary
    # print(sentiment_scores)

    sentiment_data = sentiment_scores

    # Normalize sentiment scores (assuming values range from 0 to 1)
    def normalize_sentiment(sentiment_dict):
        for brand, features in sentiment_dict.items():
            max_value = max(features.values())
            for feature, sentiment in features.items():
                sentiment_dict[brand][feature] = sentiment / max_value
        return sentiment_dict

    sentiment_data = normalize_sentiment(sentiment_data.copy())  # Normalize on a copy

    # Extract feature labels and sentiment scores for each brand
    brands = list(sentiment_data.keys())
    feature_labels = list(sentiment_data[brands[0]].keys())
    brand1_sentiment = [sentiment_data[brand1[0].upper()+brand1[1:]][feature] for feature in feature_labels]
    brand2_sentiment = [sentiment_data[brand2[0].upper()+brand2[1:]][feature] for feature in feature_labels]

    # Create a bar chart
    barfig=plt.figure()
    x = range(len(feature_labels))
    bar_width = 0.35
    plt.bar(x, brand1_sentiment, bar_width, label=brand1[0].upper()+brand1[1:])
    plt.bar([p + bar_width for p in x], brand2_sentiment, bar_width, label='Samsung')
    plt.xlabel('Feature')
    plt.ylabel('Normalized Sentiment Score (0-1)')
    title = 'Comparison of Average Sentiment by Feature (Normalized)'
    plt.title(title)
    plt.xticks([p + bar_width / 2 for p in x], feature_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    buff=io.BytesIO()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plot_url = base64.b64encode(buff.getvalue()).decode('utf8')
    jsonobjs["bar4"]=plot_url
    gfig=plt.figure()
    G = nx.Graph()

    # Add brands and features as nodes
    brands = sentiment_scores.keys()
    features = list(sentiment_scores[brand1[0].upper()+brand1[1:]].keys())  # Assuming features are the same across brands

    for brand in brands:
        G.add_node(brand, type="brand")  

    for feature in features:
        G.add_node(feature, type="feature") 

    sentiment_threshold = 0.01  # Example threshold

    for brand, feature_data in sentiment_scores.items():
        for feature, sentiment in feature_data.items():
            if sentiment >= sentiment_threshold: 
                G.add_edge(brand, feature, weight=sentiment)

    # Node Importance
    degree_centrality = nx.degree_centrality(G)  

    for brand in brands:
        brand_features = [n for n in G.neighbors(brand)]  # Features connected to a brand
        top_features = sorted(brand_features, key=lambda x: degree_centrality[x], reverse=True)[:3]
        print(f"Top Features for {brand}: {top_features}")
        
    # Customize visualization 
    node_colors = ['lightblue' if G.nodes[n]['type'] == 'brand' else 'orange' for n in G.nodes()]
    edge_widths = [d['weight'] * 3 for (_, _, d) in G.edges(data=True)]  # Adjust for visual clarity
    pos = nx.spring_layout(G)  # You can experiment with different layouts

    nx.draw(G, pos, with_labels=True, node_color=node_colors, width=edge_widths)
    buff=io.BytesIO()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plot_url = base64.b64encode(buff.getvalue()).decode('utf8')
    jsonobjs["bar5"]=plot_url

    # Check if 'day', 'brand1_sentiment', and 'brand2_sentiment' columns exist
    if all(col in dfr.columns for col in ['day', 'brand1_sentiment', 'brand2_sentiment']):
        # Convert date strings to datetime format
        dfr['day'] = pd.to_datetime(dfr['day'], format='%d-%m-%Y')

        # Normalize sentiment scores (0-1 range)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        dfr[['brand1_sentiment', 'brand2_sentiment']] = scaler.fit_transform(dfr[['brand1_sentiment', 'brand2_sentiment']])

        # Plot sentiment trends (corrected)
        overtime=plt.figure()
        dfr.plot(x='day', y=['brand1_sentiment', 'brand2_sentiment']) 
        plt.xlabel('Date')
        plt.ylabel('Normalized Sentiment')
        plt.title('Sentiment Trend Analysis')
        plt.grid(True)
        plt.legend()

        buff=io.BytesIO()
        plt.savefig(buff,format='png')
        buff.seek(0)
        plot_url = base64.b64encode(buff.getvalue()).decode('utf8')
        jsonobjs["bar6"]=plot_url
        print(overtime)
    else:
        print(f"Error: The data file  is missing required columns.")
    jsonobjs["branda"]=brand1
    jsonobjs["brandb"]=brand2
    return jsonobjs
