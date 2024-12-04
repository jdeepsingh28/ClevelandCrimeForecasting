from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from data_pipelining import run_pipeline
from datetime import datetime

app = FastAPI()

# Set up the templates folder
templates = Jinja2Templates(directory="templates")
userTemplates = Jinja2Templates(directory="templates/userTemplates")
modelTemplates = Jinja2Templates(
    directory="templates/userTemplates/modelTemplates")
devTemplates = Jinja2Templates(directory="templates/devTemplates")
# Create more templates to allow for simplification and grouping of html files

# Load the data
data = pd.read_csv("crime_weather_preprocessed.csv")

# Home route


@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Route for the user page


@app.get("/user", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("user.html", {"request": request})

# Route for the developer page


@app.get("/developer", response_class=HTMLResponse)
async def developer_page(request: Request):
    return templates.TemplateResponse("developer.html", {"request": request})


# For the analysis page put all routes here
# -----------------------------------------------------------------------
# Weather and crime map route
@app.get("/weather-crime-map", response_class=HTMLResponse)
async def weather_crime_map(request: Request):
    return userTemplates.TemplateResponse("weather_crime_map_by_census_block_with_time.html", {"request": request})


@app.get("/crime-analysis", response_class=HTMLResponse)
# Route to display powerBI
async def crime_PBI(request: Request):
    return userTemplates.TemplateResponse("powerBI.html", {"request": request})


@app.get("/crime-analysis-weather", response_class=HTMLResponse)
# Route to display powerBI
async def weather_PBI(request: Request):
    return userTemplates.TemplateResponse("powerBI_weather.html", {"request": request})


@app.get("/crime-analysis-sports", response_class=HTMLResponse)
# Route to display powerBI
async def sports_PBI(request: Request):
    return userTemplates.TemplateResponse("powerBI_sport.html", {"request": request})


@app.get("/neuralnet", response_class=HTMLResponse)
async def neural_net(request: Request):
    # Load the CSV file
    # Replace with your CSV file path
    nn_data = pd.read_csv(
        "/Users/nharms/Documents/College/CS/Senior_Project/front_end/templates/userTemplates/modelTemplates/NNVisuals/one_week_ahead_predictions.csv")

    # Convert the DataFrame to an HTML table
    # Add classes for styling (e.g., Bootstrap)
    html_table = nn_data.to_html(index=False, classes="table table-striped")

    # Render the neural network page and pass the table
    return modelTemplates.TemplateResponse("nn.html", {
        "request": request,
        "table": html_table
    })


@app.get("/xgboost", response_class=HTMLResponse)
# Route to display powerBI
async def xg_boost(request: Request):
    feature_importance = pd.read_csv(
        "/Users/nharms/Documents/College/CS/Senior_Project/front_end/templates/userTemplates/modelTemplates/XGBVisuals/feature_importances.csv")

    # Convert the DataFrame to an HTML table
    # Add classes for styling (e.g., Bootstrap)
    html_table1 = feature_importance.to_html(
        index=False, classes="table table-striped")

    predictions = pd.read_csv(
        "/Users/nharms/Documents/College/CS/Senior_Project/front_end/templates/userTemplates/modelTemplates/XGBVisuals/future_data_preds.csv")

    # Convert the DataFrame to an HTML table
    # Add classes for styling (e.g., Bootstrap)
    html_table2 = predictions.to_html(
        index=False, classes="table table-striped")

    # Render the neural network page and pass the table
    return modelTemplates.TemplateResponse("xgboost.html", {
        "request": request,
        "table1": html_table1,
        "table2": html_table2
    })


# Route to display the data page


@app.get("/data", response_class=HTMLResponse)
async def display_data(request: Request):
    return userTemplates.TemplateResponse("data.html", {"request": request})


# Paginated data endpoint


@app.get("/api/data", response_class=JSONResponse)
async def get_data(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    start = (page - 1) * page_size
    end = start + page_size
    data_chunk = data.iloc[start:end].to_dict(orient="records")
    return {
        "data": data_chunk,
        "total": len(data)
    }

# Route to display the models page


@app.get("/models", response_class=HTMLResponse)
async def models(request: Request):
    return userTemplates.TemplateResponse("models.html", {"request": request})

# ------------------------------------------------------------------------


# For developer page put all routes here
# ------------------------------------------------------------------------

# Route for the data pipelining page


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    return devTemplates.TemplateResponse("pipeline.html", {"request": request})

# Runs the data pipeline in data_pipelining file


@app.post("/run-pipeline", response_class=HTMLResponse)
async def run_pipeline_route(request: Request, start_date: str = Form(...), end_date: str = Form(...)):
    # Convert form data to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Run the pipelining function with the parsed dates
    result = run_pipeline(start_date_obj, end_date_obj)

    # Return a response showing the result
    return devTemplates.TemplateResponse("pipeline_result.html", {"request": request, "message": result})


@app.get("/devModels", response_class=HTMLResponse)
async def devModels(request: Request):
    return devTemplates.TemplateResponse("devModels.html", {"request": request})

# ------------------------------------------------------------------------
