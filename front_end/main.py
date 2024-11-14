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


@app.get("/analysis", response_class=HTMLResponse)
# Route to display powerBI
async def developer_page(request: Request):
    return userTemplates.TemplateResponse("powerBI.html", {"request": request})


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
