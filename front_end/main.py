#This was slapped together pretty quickly so far and needs a lot of work to be good. 


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

# Load the data
data = pd.read_csv("crime_weather_preprocessed.csv")

# Home route
@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Weather and crime map route
@app.get("/weather-crime-map", response_class=HTMLResponse)
async def weather_crime_map(request: Request):
    return templates.TemplateResponse("weather_crime_map_by_census_block_with_time.html", {"request": request})


# Route to display the data page
@app.get("/data", response_class=HTMLResponse)
async def display_data(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})

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
    
@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    return templates.TemplateResponse("pipeline.html", {"request": request})


@app.post("/run-pipeline", response_class=HTMLResponse)
async def run_pipeline_route(request: Request, start_date: str = Form(...), end_date: str = Form(...)):
    # Convert form data to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Run the pipelining function with the parsed dates
    result = run_pipeline(start_date_obj, end_date_obj)

    # Return a response showing the result
    return templates.TemplateResponse("pipeline_result.html", {"request": request, "message": f"Pipeline processed {result} records between {start_date} and {end_date}."})