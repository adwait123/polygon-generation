import streamlit as st
import pandas as pd
import numpy as np
import json
import math
import time
import io
import zipfile
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

# Page config
st.set_page_config(
    page_title="ZIP Code Polygon Generator",
    page_icon="📍",
    layout="wide"
)

st.title("📍 ZIP Code Polygon Generator")
st.markdown("Generate optimal service area polygons using Graham Scan convex hull algorithm")


def graham_scan(points):
    """Graham scan algorithm to find convex hull of points."""
    if len(points) < 3:
        return points

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def polar_angle(p0, p1):
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def distance_squared(p0, p1):
        return (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2

    start = min(points, key=lambda p: (p[1], p[0]))

    def compare_points(p):
        angle = polar_angle(start, p)
        dist = distance_squared(start, p)
        return (angle, dist)

    sorted_points = sorted([p for p in points if p != start], key=compare_points)

    if len(sorted_points) == 0:
        return [start]

    hull = [start, sorted_points[0]]

    for p in sorted_points[1:]:
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def create_optimal_polygons(assignments_df, zip_db_df):
    """Direct port of your original function with progress tracking."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Setting up coordinate lookup...")

    # Create optimized lookup - exactly like your original
    zip_coordinates = {}
    for _, row in zip_db_df.iterrows():
        zip_int = int(row['zip'])
        zip_coordinates[zip_int] = (row['lat'], row['lng'])

    status_text.text("Grouping assignments by business...")

    # Group assignments by business - exactly like your original
    business_groups = {}
    for _, row in assignments_df.iterrows():
        business_id = row['business_id']
        biz_zip = int(row['biz_zip'])

        if business_id not in business_groups:
            business_groups[business_id] = {
                'business_id': business_id,
                'biz_zip': biz_zip,
                'assignments': []
            }

        business_groups[business_id]['assignments'].append({
            'zip': int(row['target_zip']),
            'distance': row['distance_miles']
        })

    # Optimized distance calculation with caching - exactly like your original
    distance_cache = {}

    def cached_distance(zip1, zip2, debug=False):
        cache_key = (min(zip1, zip2), max(zip1, zip2))
        if cache_key in distance_cache:
            return distance_cache[cache_key]

        if zip1 not in zip_coordinates or zip2 not in zip_coordinates:
            return float('inf')

        coords1 = zip_coordinates[zip1]
        coords2 = zip_coordinates[zip2]
        distance = geodesic(coords1, coords2).miles
        distance_cache[cache_key] = distance
        return distance

    def validate_distance_constraint(zips, max_distance=85, debug=False):
        for i in range(len(zips)):
            if zips[i] not in zip_coordinates:
                continue
            for j in range(i + 1, len(zips)):
                if zips[j] not in zip_coordinates:
                    continue
                distance = cached_distance(zips[i], zips[j], debug)
                if distance > max_distance:
                    violation = {'zip1': zips[i], 'zip2': zips[j], 'distance': distance}
                    return False, violation
        return True, None

    def count_contained_zips(polygon_zips, all_zips, debug=False):
        if len(polygon_zips) < 3:
            return 0, []

        polygon_points = []
        for zip_code in polygon_zips:
            if zip_code in zip_coordinates:
                lng, lat = zip_coordinates[zip_code][1], zip_coordinates[zip_code][0]
                polygon_points.append((lng, lat))

        if len(polygon_points) < 3:
            return 0, []

        try:
            polygon = Polygon(polygon_points)
            if not polygon.is_valid:
                try:
                    from shapely.validation import make_valid
                    polygon = make_valid(polygon)
                except ImportError:
                    polygon_points.reverse()
                    polygon = Polygon(polygon_points)
        except Exception:
            return 0, []

        contained_zips = []
        minx, miny, maxx, maxy = polygon.bounds

        # Bounding box optimization - exactly like your original
        potential_zips = []
        for zip_code in all_zips:
            if zip_code not in zip_coordinates:
                continue
            lng, lat = zip_coordinates[zip_code][1], zip_coordinates[zip_code][0]
            if minx <= lng <= maxx and miny <= lat <= maxy:
                potential_zips.append(zip_code)

        for zip_code in potential_zips:
            lng, lat = zip_coordinates[zip_code][1], zip_coordinates[zip_code][0]
            point = Point(lng, lat)
            if polygon.contains(point) or polygon.touches(point):
                contained_zips.append(zip_code)

        return len(contained_zips), contained_zips

    def create_constrained_convex_hull(business_zip, valid_zips, max_distance=85, max_points=6):
        min_points = 3
        if len(valid_zips) < min_points:
            return [], []

        # Sampling logic - exactly like your original
        sample_size = 500
        working_zips = valid_zips
        if len(valid_zips) > sample_size:
            if business_zip in valid_zips and business_zip in zip_coordinates:
                valid_zips_without_business = [z for z in valid_zips if z != business_zip]
                if len(valid_zips_without_business) > sample_size - 1:
                    import random
                    random.seed(42)
                    sampled_zips = random.sample(valid_zips_without_business, sample_size - 1)
                    sampled_zips.append(business_zip)
                else:
                    sampled_zips = valid_zips
            else:
                import random
                random.seed(42)
                sampled_zips = random.sample(valid_zips, min(sample_size, len(valid_zips)))
            working_zips = sampled_zips

        valid_zip_coords = [z for z in working_zips if z in zip_coordinates]

        # Build distance-constrained clusters - exactly like your original
        clusters = []
        for center_zip in valid_zip_coords:
            cluster = [center_zip]
            for other_zip in valid_zip_coords:
                if other_zip == center_zip:
                    continue

                valid_for_cluster = True
                for cluster_zip in cluster:
                    if cached_distance(other_zip, cluster_zip) > max_distance:
                        valid_for_cluster = False
                        break

                if valid_for_cluster:
                    cluster.append(other_zip)

            if len(cluster) >= min_points:
                clusters.append(cluster)

        if not clusters:
            return [], []

        best_hull_zips = []
        best_contained_zips = []
        best_coverage = 0

        # Evaluate clusters - exactly like your original (top 10)
        for cluster in clusters[:10]:
            cluster_points = []
            zip_to_point = {}
            for zip_code in cluster:
                if zip_code in zip_coordinates:
                    lat, lng = zip_coordinates[zip_code]
                    point = (lng, lat)
                    cluster_points.append(point)
                    zip_to_point[point] = zip_code

            if len(cluster_points) < min_points:
                continue

            hull_points = graham_scan(cluster_points)

            # Limit hull points to max_points - exactly like your original
            if len(hull_points) > max_points:
                best_subset_coverage = 0
                best_subset_hull = []
                best_subset_contained = []

                from itertools import combinations

                if len(hull_points) > 12:
                    step = len(hull_points) // max_points
                    hull_subset = [hull_points[i] for i in range(0, len(hull_points), step)][:max_points]
                    hull_combinations = [hull_subset]
                else:
                    hull_combinations = list(combinations(hull_points, max_points))[:50]

                for hull_subset in hull_combinations:
                    subset_hull_zips = [zip_to_point[point] for point in hull_subset]
                    is_valid, _ = validate_distance_constraint(subset_hull_zips, max_distance)
                    if not is_valid:
                        continue

                    coverage_count, contained = count_contained_zips(subset_hull_zips, valid_zips)
                    if coverage_count > best_subset_coverage:
                        best_subset_coverage = coverage_count
                        best_subset_hull = subset_hull_zips
                        best_subset_contained = contained

                hull_zips = best_subset_hull
                coverage_count = best_subset_coverage
                contained = best_subset_contained

                if not hull_zips:
                    continue
            else:
                hull_zips = [zip_to_point[point] for point in hull_points]
                is_valid, _ = validate_distance_constraint(hull_zips, max_distance)
                if not is_valid:
                    continue
                coverage_count, contained = count_contained_zips(hull_zips, valid_zips)

            if coverage_count > best_coverage:
                best_coverage = coverage_count
                best_hull_zips = hull_zips
                best_contained_zips = contained

        return best_hull_zips, best_contained_zips

    # Process each business - exactly like your original
    results = []
    covered_zips_data = []
    uncovered_zips_data = []

    total_businesses = len(business_groups)

    for i, (business_id, group) in enumerate(business_groups.items()):
        progress = (i + 1) / total_businesses
        progress_bar.progress(progress)
        status_text.text(f"Processing business {i + 1}/{total_businesses}: {business_id}")

        business_zip = group['biz_zip']
        all_assigned_zips = [a['zip'] for a in group['assignments']]

        # Filter valid ZIPs - exactly like your original
        valid_zips = []
        business_coords = zip_coordinates.get(business_zip)

        for assignment in group['assignments']:
            zip_code = assignment['zip']
            if zip_code in zip_coordinates:
                is_valid = False
                if business_coords:
                    distance = geodesic(business_coords, zip_coordinates[zip_code]).miles
                    is_valid = distance <= 85
                else:
                    is_valid = assignment['distance'] <= 85

                if is_valid:
                    valid_zips.append(zip_code)

        # Create polygon - exactly like your original logic
        min_points = 3
        if len(valid_zips) >= min_points:
            optimal_polygon, contained_zips = create_constrained_convex_hull(
                business_zip, valid_zips, max_distance=85, max_points=6
            )

            if len(optimal_polygon) >= min_points:
                is_valid, _ = validate_distance_constraint(optimal_polygon, max_distance=85)

                if is_valid:
                    coverage_percentage = round((len(contained_zips) / len(valid_zips)) * 100) if valid_zips else 0

                    # Track covered ZIPs
                    for zip_code in contained_zips:
                        covered_zips_data.append({
                            'business_id': business_id,
                            'biz_zip': business_zip,
                            'zip_code': zip_code,
                            'polygon_boundary': ';'.join(map(str, optimal_polygon))
                        })

                    # Track uncovered ZIPs
                    uncovered_zips = [zip_code for zip_code in all_assigned_zips if zip_code not in contained_zips]
                    for zip_code in uncovered_zips:
                        reason = 'Outside convex hull boundary'
                        if zip_code not in valid_zips:
                            if zip_code not in zip_coordinates:
                                reason = 'No coordinates available'
                            else:
                                reason = 'Outside 85-mile distance constraint'

                        uncovered_zips_data.append({
                            'business_id': business_id,
                            'biz_zip': business_zip,
                            'zip_code': zip_code,
                            'reason': reason
                        })

                    results.append({
                        'business_id': business_id,
                        'biz_zip': business_zip,
                        'boundary_zips': ';'.join(map(str, optimal_polygon)),
                        'contained_zip_count': len(contained_zips),
                        'contained_zips': contained_zips,
                        'total_assigned_zips': len(all_assigned_zips),
                        'within_distance_constraint': len(valid_zips),
                        'coverage_percentage': coverage_percentage,
                        'has_valid_polygon': "Yes",
                        'polygon_point_count': len(optimal_polygon)
                    })
                    continue

        # No valid polygon case
        for zip_code in all_assigned_zips:
            uncovered_zips_data.append({
                'business_id': business_id,
                'biz_zip': business_zip,
                'zip_code': zip_code,
                'reason': 'Could not create valid polygon'
            })

        results.append({
            'business_id': business_id,
            'biz_zip': business_zip,
            'boundary_zips': "",
            'contained_zip_count': 0,
            'contained_zips': [],
            'total_assigned_zips': len(all_assigned_zips),
            'within_distance_constraint': len(valid_zips),
            'coverage_percentage': 0,
            'has_valid_polygon': "No",
            'polygon_point_count': 0
        })

    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    return results, covered_zips_data, uncovered_zips_data


def create_visualization_html(polygons_df, zip_coordinates, assignments_df):
    """Create HTML visualization using folium - matches original script output."""
    try:
        import folium
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from branca.element import Element
        from shapely.geometry import MultiPoint
    except ImportError:
        st.error("Folium not installed. Cannot create visualization.")
        return None

    # Center map
    all_coords = list(zip_coordinates.values())
    mean_lat = sum([c[0] for c in all_coords]) / len(all_coords)
    mean_lng = sum([c[1] for c in all_coords]) / len(all_coords)
    m = folium.Map(location=[mean_lat, mean_lng], zoom_start=5, tiles='cartodbpositron')

    # Calculate summary stats
    total_covered = polygons_df['contained_zip_count'].sum()
    total_assigned = polygons_df['total_assigned_zips'].sum()
    total_coverage_pct = (total_covered / total_assigned * 100) if total_assigned else 0

    # Add summary HTML to map - exactly like original
    summary_html = f'''
    <div style="position: fixed; z-index: 9999; background: white; padding: 10px; border: 2px solid #333; top: 10px; left: 50%; transform: translateX(-50%); font-size: 16px;">
        <b>Total Coverage:</b> {total_coverage_pct:.1f}%<br>
        <b>ZIPs covered:</b> {total_covered} / {total_assigned}<br>
        <b>Algorithm:</b> Graham Scan Convex Hull (Max 6 Points)<br>
        <div style="margin-top: 8px; font-size: 14px;">
            <span style="color: yellow; background: black; padding: 2px;">●</span> Covered ZIPs &nbsp;
            <span style="color: darkred;">●</span> Uncovered ZIPs &nbsp;
            <span style="color: red;">✕</span> Unassigned ZIPs
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(summary_html))

    # Load all assigned zips for each business from assignments data
    assignments = {}
    for _, row in assignments_df.iterrows():
        biz = row['business_id']
        if biz not in assignments:
            assignments[biz] = set()
        assignments[biz].add(int(row['target_zip']))

    # Draw polygons with full visualization - exactly like original
    for _, row in polygons_df.iterrows():
        if not isinstance(row['boundary_zips'], str) or not row['boundary_zips']:
            continue

        business_id = row['business_id']
        # Split on semicolons for Graham Scan output
        boundary_zips = [int(z) for z in str(row['boundary_zips']).split(';') if z.strip().isdigit()]

        contained_zips = []
        # Handle contained_zips parsing - exactly like original
        if 'contained_zips' in row:
            if isinstance(row['contained_zips'], str) and row['contained_zips'].strip():
                # String format: "[12345, 12346, 12347]" or "12345,12346,12347"
                contained_zips_str = row['contained_zips'].strip('[]').replace("'", "").replace('"', '')
                contained_zips = [int(z.strip()) for z in contained_zips_str.split(',') if z.strip().isdigit()]
            elif isinstance(row['contained_zips'], list):
                # Already a list
                contained_zips = [int(z) for z in row['contained_zips'] if str(z).isdigit()]
            elif pd.notna(row['contained_zips']):
                # Try to evaluate if it looks like a list string
                try:
                    import ast
                    contained_zips = ast.literal_eval(str(row['contained_zips']))
                    if isinstance(contained_zips, list):
                        contained_zips = [int(z) for z in contained_zips if str(z).isdigit()]
                    else:
                        contained_zips = []
                except:
                    contained_zips = []

        # Uncovered zips: assigned but not contained
        assigned_zips = assignments.get(business_id, set())
        uncovered_zips = [z for z in assigned_zips if z not in contained_zips and z in zip_coordinates]

        # Draw uncovered zips as red dots - exactly like original
        for z in uncovered_zips:
            folium.CircleMarker(
                location=zip_coordinates[z],
                radius=4,
                color='darkred',
                fill=True,
                fill_color='red',
                fill_opacity=0.9,
                weight=1,
                tooltip=f"Uncovered ZIP: {z}"
            ).add_to(m)

        # Draw convex hull polygon in orange - exactly like original
        poly_points = [zip_coordinates[z] for z in boundary_zips if z in zip_coordinates]
        if len(poly_points) >= 3:
            try:
                # Convert to (lng, lat) for shapely
                shapely_points = [(lng, lat) for lat, lng in poly_points]
                hull = MultiPoint(shapely_points).convex_hull
                if hull.geom_type == 'Polygon':
                    hull_points = [(lat, lng) for lng, lat in hull.exterior.coords]
                    folium.Polygon(
                        locations=hull_points,
                        color='orange',
                        fill=True,
                        fill_opacity=0.15,
                        weight=2,
                        tooltip=f"Convex Hull for Business {business_id}"
                    ).add_to(m)
            except Exception as e:
                pass  # Skip if error drawing convex hull

        # Draw polygon with popup showing boundary ZIPs - exactly like original
        if len(poly_points) >= 3:
            popup_html = f"<b>Boundary ZIPs:</b> {'; '.join(map(str, boundary_zips))}<br><b>Points:</b> {len(boundary_zips)}/6"
            folium.Polygon(
                locations=poly_points,
                color='blue',
                fill=True,
                fill_opacity=0.2,
                weight=2,
                tooltip=f"Business {business_id}",
                popup=folium.Popup(popup_html, max_width=350)
            ).add_to(m)

        # Draw contained zips as small GREEN dots - exactly like original
        for z in contained_zips:
            if z in zip_coordinates:
                folium.CircleMarker(
                    location=zip_coordinates[z],
                    radius=3,  # Increased from 2 to 3
                    color='darkgreen',  # Changed from gray to dark green
                    fill=True,
                    fill_color='lightgreen',  # Added fill color
                    fill_opacity=0.8,  # Increased opacity
                    weight=1,  # Added border weight
                    tooltip=f"Covered ZIP: {z}"  # Added tooltip
                ).add_to(m)

        # Draw business ZIP as a green marker with popup - exactly like original
        if int(row['biz_zip']) in zip_coordinates:
            popup_html = f"""
            <b>Business ID:</b> {business_id}<br>
            <b>Business ZIP:</b> {row['biz_zip']}<br>
            <b>Boundary ZIPs:</b> {'; '.join(map(str, boundary_zips))}<br>
            <b>Polygon Points:</b> {len(boundary_zips)}/6<br>
            <b>Convex Hull drawn in orange</b><br>
            <b>Coverage:</b> {row.get('coverage_percentage', 0)}%
            """
            folium.Marker(
                location=zip_coordinates[int(row['biz_zip'])],
                icon=folium.Icon(color='green', icon='briefcase'),
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"Business ZIP: {row['biz_zip']}"
            ).add_to(m)

    return m._repr_html_()


# File upload section
st.header("📁 Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ZIP Code Assignments")
    assignments_file = st.file_uploader(
        "Upload assignments CSV",
        type=['csv'],
        help="CSV with columns: business_id, biz_zip, target_zip, distance_miles"
    )

with col2:
    st.subheader("ZIP Code Database")
    zip_db_file = st.file_uploader(
        "Upload ZIP database CSV",
        type=['csv'],
        help="CSV with columns: zip, lat, lng"
    )

# Process data when both files are uploaded
if assignments_file and zip_db_file:
    try:
        # Load data
        assignments_df = pd.read_csv(assignments_file)
        zip_db_df = pd.read_csv(zip_db_file)

        # Validate columns
        required_assignments_cols = ['business_id', 'biz_zip', 'target_zip', 'distance_miles']
        required_zip_cols = ['zip', 'lat', 'lng']

        missing_assignments = [col for col in required_assignments_cols if col not in assignments_df.columns]
        missing_zip = [col for col in required_zip_cols if col not in zip_db_df.columns]

        if missing_assignments or missing_zip:
            st.error("Missing required columns:")
            if missing_assignments:
                st.error(f"Assignments file missing: {missing_assignments}")
            if missing_zip:
                st.error(f"ZIP database missing: {missing_zip}")
        else:
            # Show data preview
            st.header("📊 Data Preview")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Assignments Data")
                st.dataframe(assignments_df.head())
                st.write(f"Total rows: {len(assignments_df):,}")
                st.write(f"Unique businesses: {assignments_df['business_id'].nunique():,}")

            with col2:
                st.subheader("ZIP Database")
                st.dataframe(zip_db_df.head())
                st.write(f"Total ZIP codes: {len(zip_db_df):,}")

            # Process button
            if st.button("🚀 Generate Polygons", type="primary"):
                # Store results in session state to persist across reruns
                st.session_state.processing_complete = False

                st.header("⚙️ Processing")

                start_time = time.time()

                with st.spinner("Processing data..."):
                    results, covered_zips_data, uncovered_zips_data = create_optimal_polygons(assignments_df, zip_db_df)

                end_time = time.time()
                processing_time = round(end_time - start_time, 2)

                # Store all results in session state
                st.session_state.results = results
                st.session_state.covered_zips_data = covered_zips_data
                st.session_state.uncovered_zips_data = uncovered_zips_data
                st.session_state.processing_time = processing_time
                st.session_state.assignments_df = assignments_df
                st.session_state.zip_db_df = zip_db_df
                st.session_state.processing_complete = True

            # Display results if processing is complete
            if st.session_state.get('processing_complete', False):
                # Retrieve results from session state
                results = st.session_state.results
                covered_zips_data = st.session_state.covered_zips_data
                uncovered_zips_data = st.session_state.uncovered_zips_data
                processing_time = st.session_state.processing_time
                assignments_df = st.session_state.assignments_df
                zip_db_df = st.session_state.zip_db_df

                # Create output DataFrames
                polygons_df = pd.DataFrame([
                    {
                        'business_id': r['business_id'],
                        'biz_zip': r['biz_zip'],
                        'boundary_zips': r['boundary_zips'],
                        'contained_zip_count': r['contained_zip_count'],
                        'contained_zips': str(r['contained_zips']) if r['contained_zips'] else "[]",
                        'total_assigned_zips': r['total_assigned_zips'],
                        'within_distance_constraint': r['within_distance_constraint'],
                        'coverage_percentage': r['coverage_percentage'],
                        'has_valid_polygon': r['has_valid_polygon'],
                        'polygon_point_count': r['polygon_point_count']
                    }
                    for r in results
                ])

                covered_df = pd.DataFrame(covered_zips_data) if covered_zips_data else pd.DataFrame()
                uncovered_df = pd.DataFrame(uncovered_zips_data) if uncovered_zips_data else pd.DataFrame()

                # Results summary
                st.header("📈 Results Summary")

                valid_polygons = sum(1 for r in results if r['has_valid_polygon'] == "Yes")
                total_covered = len(covered_zips_data)
                total_uncovered = len(uncovered_zips_data)
                total_zips = total_covered + total_uncovered
                coverage_rate = (total_covered / total_zips * 100) if total_zips > 0 else 0

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total Businesses", len(results))

                with col2:
                    st.metric("Valid Polygons", valid_polygons)

                with col3:
                    st.metric("Coverage Rate", f"{coverage_rate:.1f}%")

                with col4:
                    st.metric("Covered ZIPs", f"{total_covered:,}")

                with col5:
                    st.metric("Processing Time", f"{processing_time}s")

                # Display results
                st.subheader("Polygon Results")
                st.dataframe(polygons_df, use_container_width=True)

                # Create visualization
                st.header("🗺️ Visualization")
                with st.spinner("Creating map visualization..."):
                    try:
                        # Create coordinate lookup for visualization
                        zip_coordinates = dict(zip(zip_db_df['zip'].astype(int),
                                                   zip(zip_db_df['lat'], zip_db_df['lng'])))

                        html_content = create_visualization_html(polygons_df, zip_coordinates, assignments_df)

                        if html_content:
                            st.components.v1.html(html_content, height=600)
                        else:
                            st.warning("Could not create visualization. Folium library may not be available.")
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")

                # Download section
                st.header("📥 Download Results")


                # Prepare downloads
                def to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')


                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📊 Download Polygons CSV",
                        data=to_csv(polygons_df),
                        file_name="zip_code_polygons.csv",
                        mime="text/csv"
                    )

                    if not covered_df.empty:
                        st.download_button(
                            label="✅ Download Covered ZIPs CSV",
                            data=to_csv(covered_df),
                            file_name="covered_zips.csv",
                            mime="text/csv"
                        )

                with col2:
                    if not uncovered_df.empty:
                        st.download_button(
                            label="❌ Download Uncovered ZIPs CSV",
                            data=to_csv(uncovered_df),
                            file_name="uncovered_zips.csv",
                            mime="text/csv"
                        )

                    # HTML visualization download
                    if 'html_content' in locals() and html_content:
                        st.download_button(
                            label="🗺️ Download Map HTML",
                            data=html_content.encode('utf-8'),
                            file_name="polygon_map.html",
                            mime="text/html"
                        )

                # Create ZIP bundle
                if st.button("📦 Download All Files as ZIP"):
                    zip_buffer = io.BytesIO()

                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Add CSV files
                        zip_file.writestr("zip_code_polygons.csv", polygons_df.to_csv(index=False))

                        if not covered_df.empty:
                            zip_file.writestr("covered_zips.csv", covered_df.to_csv(index=False))

                        if not uncovered_df.empty:
                            zip_file.writestr("uncovered_zips.csv", uncovered_df.to_csv(index=False))

                        # Add HTML if available
                        if 'html_content' in locals() and html_content:
                            zip_file.writestr("polygon_map.html", html_content)

                    st.download_button(
                        label="📦 Download ZIP Bundle",
                        data=zip_buffer.getvalue(),
                        file_name="polygon_results.zip",
                        mime="application/zip"
                    )

                # Add a reset button to clear results and start over
                if st.button("🔄 Process New Files"):
                    for key in ['processing_complete', 'results', 'covered_zips_data', 'uncovered_zips_data',
                                'processing_time', 'assignments_df', 'zip_db_df']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    except Exception as e:
        st.error(f"Error loading files: {str(e)}")

else:
    st.info("👆 Please upload both CSV files to begin processing")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application generates optimal service area polygons.

    **Required Files:**
    1. **Assignments CSV**: business_id, biz_zip, target_zip, distance_miles
    2. **ZIP Database CSV**: zip, lat, lng

    **Outputs:**
    1. Polygon boundaries CSV
    2. Covered ZIPs CSV  
    3. Uncovered ZIPs CSV
    4. Interactive HTML map
    """)
