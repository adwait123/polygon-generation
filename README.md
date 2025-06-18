# ZIP Code Polygon Generator

A Streamlit application that generates optimal ZIP code polygons using the Graham Scan convex hull algorithm.

## Features
- Upload CSV files with ZIP code assignments
- Generate convex hull polygons with distance constraints
- Interactive visualizations
- Export results in multiple formats

## Usage
1. Upload your ZIP code assignments CSV
2. Upload your ZIP code database CSV
3. Configure settings in the sidebar
4. Click "Generate Polygons"
5. Download your results

## File Formats
- **Assignments CSV**: business_id, biz_zip, target_zip, distance_miles
- **ZIP Database CSV**: zip, lat, lng
