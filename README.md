# 🏎️ F1 Prediction 3000

Predict Formula 1 race outcomes using Machine Learning, historical data, and live FastF1 practice/qualifying sessions! 

**Credits & Origin:**
- 🧠 **Original Core Engine Structure & ML Codebase by:** [@mar_antaya](https://www.instagram.com/mar_antaya) (Check them out on Instagram and TikTok for predictions before every F1 race!)
- ⚡ **2026 Season Automation & Streamlit App Features by:** [@wild__zee](https://www.instagram.com/wild__zee) on Instagram.

---

## 🚀 What This Does
This app uses a Machine Learning model (**XGBoost**) to predict the full 22-driver grid results for the upcoming 2026 Formula 1 season. 

**Awesome Features added for 2026:**
- **Fully Automated:** It hooks directly into the `FastF1` API to grab live Practice (FP1/FP2/FP3) and Qualifying data automatically on race weekends.
- **Qualifying Predictor:** Uses live practice pace blended with historical data to predict pole position.
- **Race Predictor:** Predicts the race winner based on qualifying times, weather forecasts, and historical track pace.
- **Beautiful UI:** A sleek, fully interactive Streamlit web dashboard.

---

## 🛠️ How to Start (For Beginners!)

Even if you're a complete "noob" at coding, you can run this app easily on your computer by following these steps.

### Step 1: Install Python
You need Python installed on your computer. Download and install it from [python.org](https://www.python.org/downloads/). 

> **Important for Windows Users:** During the Python installation, make sure you check the box that says **"Add Python to PATH"** before clicking Install!

### Step 2: Download This Code
Download this folder to your computer (or clone the repository if you know how to use Git), and open your terminal (Mac/Linux) or Command Prompt / PowerShell (Windows).

Use the `cd` command to navigate into the folder where you saved this code:
```bash
cd path/to/2025_f1_predictions
```
*(The folder might still be named 2025, but the app inside is ready for 2026!)*

### Step 3: Install the Required Packages
Tell Python to download the libraries needed to run the app (like `pandas`, `xgboost`, and `streamlit`):
```bash
pip install -r requirements.txt
```
*(Mac Users: Use `pip3` instead of `pip` if you get an error).*

### Step 4: Run the App!
Start the F1 dashboard by running this simple command:
```bash
python -m streamlit run app.py
```
*(Mac Users: Use `python3 -m streamlit run app.py`)*

� A browser window will automatically pop open with the app!

---

## 🏁 How to Use the App

The app has a sidebar with multiple pages. For the best, most accurate predictions, follow this workflow on a race weekend:

1. **Check the Weather:** On the dashboard, entering your free OpenWeatherMap API key (in a `.env` file) will pull live rain probability and temperature for the race weekend, as wet races change the predictions.
2. **Qualifying Predictor:** Once Free Practice sessions (FP1/2/3) finish, go to this page. Hit predict, and it will auto-fetch the practice times from FastF1 to predict the Qualifying results. Hit the "Save" button to save that qualifying grid!
3. **Race Predictor:** Go to this page next. It will automatically load your saved qualifying predictions (or real qualifying data if the session has actually finished) and predict the final race winner and podium!

---

## 📜 License
This project is licensed under the MIT License.

🏎️ **Start predicting F1 races like a data scientist!** 🚀
