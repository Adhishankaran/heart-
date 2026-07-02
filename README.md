# ⚽ FIFA World Cup 2026 KNN Analytics Lab

A premium, interactive Machine Learning dashboard built with **Streamlit**, **scikit-learn**, and **Plotly**. This application aggregates match-by-match statistics for 1,245 players from the FIFA World Cup 2026 dataset and provides three main analytics labs powered by **K-Nearest Neighbors (KNN)**.

---

## 🚀 Key Features

### 1. 🔍 Player Scouting & Similarity (KNN Search)
* **How it works:** Uses `NearestNeighbors` in scikit-learn.
* **Scouting Styles:** Select from *All-Round Performance*, *Attacking & Goalscoring*, *Playmaking & Creativity*, *Defending & Physicality*, or *Goalkeeping* to customize what player attributes the KNN model focuses on.
* **Radar Comparisons:** Generates interactive Plotly radar charts comparing the selected player's profile against the average of the found similar players.
* **Export Reports:** Download a CSV report of the top K similar players instantly.

### 2. 🛡️ Player Position Classifier (KNN Classifier)
* **How it works:** Trains a `KNeighborsClassifier` to predict a player's position (`Forward`, `Midfielder`, `Defender`, `Goalkeeper`) from their average technical stats.
* **Real-time Performance:** View test accuracy and interactive class probability bars.
* **2D PCA Projection:** Visualizes all 1,248 player profiles projected onto the first two Principal Components. 
* **Custom Prediction:** Move sliders to input custom match stats (goals, assists, tackles, saves) and see exactly where the custom player projects on the cluster chart as a highlighted gold star!

### 3. 📈 Player Rating Regressor (KNN Regressor)
* **How it works:** Trains a `KNeighborsRegressor` to predict the player's overall tournament rating (`player_rating`) based on 16 physical, biological, and technical performance statistics.
* **Model Validation:** Real-time calculation of R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
* **Interactive Predictions:** Input custom bio/performance specs to calculate predicted ratings, plotted on a histogram against the entire tournament's rating distribution.

---

## 💻 Local Setup & Running

1. **Clone or copy the files** to your local workspace directory.
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
4. Access the local server in your browser at `http://localhost:8501`.

---

## 🌐 How to Deploy to Streamlit Community Cloud (Streamlit Lab)

Streamlit Community Cloud is a free hosting platform for Streamlit applications. Follow these steps to host your KNN Analytics Lab online:

### Step 1: Upload Your Code to GitHub
1. Create a new repository on GitHub (e.g., `fifa-2026-knn-analytics`).
2. Add and commit the following files in your repository:
   * `app.py` (The main application code)
   * `requirements.txt` (List of dependencies)
   * `fifa_world_cup_2026_player_performance.csv` (The player performance dataset)
   * `.streamlit/config.toml` (Custom dark-theme styling settings)
3. Push your commits to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of FIFA 2026 KNN Analytics Lab"
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in. You can log in using your GitHub account.
2. Once logged in, click the **"New app"** button in the upper-right corner.
3. Fill in the deployment details:
   * **Repository:** Select your repository (e.g., `YOUR_GITHUB_USERNAME/fifa-2026-knn-analytics`).
   * **Branch:** Select `main` (or the branch you pushed to).
   * **Main file path:** Type `app.py`.
4. Click **Deploy!**

Your application will build (installing the packages listed in `requirements.txt`) and launch under a public URL that you can share!
