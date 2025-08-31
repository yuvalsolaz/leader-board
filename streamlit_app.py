import io
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Try to import SQLAlchemy dependencies
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    st.warning("SQLAlchemy support not available. Install sqlalchemy to enable database connections.")

# Try to import config
try:
    from config import POSTGRES_CONFIG, get_connection_string
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    st.warning("config.py not found. Create it with your PostgreSQL connection details.")


st.set_page_config(
	page_title="Model Inference Dashboard",
	page_icon="📊",
	layout="wide",
)


@st.cache_data(show_spinner=False)
def load_experiments_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Process DataFrame, coerce likely datetime columns, and return processed DataFrame.

	The function is cached to improve performance when re-running with the same data.
	"""
	if df.empty:
		return df

	# Try to detect and convert likely datetime columns
	for column_name in df.columns:
		series = df[column_name]
		if series.dtype == object or "date" in column_name.lower() or "time" in column_name.lower():
			parsed = pd.to_datetime(series, errors="coerce", utc=False)
			# Heuristic: if at least 30% successfully parsed, treat as datetime
			if parsed.notna().mean() >= 0.3:
				df[column_name] = parsed.dt.tz_localize(None)

	return df


@st.cache_data(show_spinner=False)
def load_experiments(csv_path: str) -> pd.DataFrame:
	"""Load experiments CSV, coerce likely datetime columns, and return DataFrame.

	The function is cached to improve performance when re-running with the same file.
	"""
	try:
		df = pd.read_csv(csv_path)
		return load_experiments_from_dataframe(df)
	except FileNotFoundError:
		st.error(f"Data file not found at: {csv_path}")
		return pd.DataFrame()
	except Exception as error:
		st.error(f"Failed to read CSV: {error}")
		return pd.DataFrame()


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
	"""Load and process uploaded CSV file."""
	try:
		df = pd.read_csv(uploaded_file)
		return load_experiments_from_dataframe(df)
	except Exception as error:
		st.error(f"Failed to read uploaded CSV: {error}")
		return pd.DataFrame()


def load_from_postgres(table_name: str = None, custom_query: str = None) -> pd.DataFrame:
    """Load data from PostgreSQL table using SQLAlchemy."""
    if not SQLALCHEMY_AVAILABLE:
        st.error("SQLAlchemy support not available. Install sqlalchemy.")
        return pd.DataFrame()
    
    if not CONFIG_AVAILABLE:
        st.error("Database configuration not available. Check config.py.")
        return pd.DataFrame()
    
    try:
        # Create SQLAlchemy engine
        connection_string = get_connection_string()
        engine = create_engine(connection_string)
        
        # Use provided table name or default from config
        table = table_name if table_name else POSTGRES_CONFIG.get('table_name', 'experiments')
        
        # Load data
        if custom_query:
            # Use custom query if provided
            df = pd.read_sql(text(custom_query), engine)
        else:
            # Use simple SELECT * FROM table
            df = pd.read_sql(f"SELECT * FROM {table}", engine)
        
        engine.dispose()
        return load_experiments_from_dataframe(df)
        
    except SQLAlchemyError as error:
        st.error(f"Database error: {error}")
        return pd.DataFrame()
    except Exception as error:
        st.error(f"Failed to connect to database: {error}")
        return pd.DataFrame()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
	if df.empty:
		return []
	categorical_cols: List[str] = []
	for column_name in df.columns:
		series = df[column_name]
		if series.dtype == object or series.dtype == bool:
			categorical_cols.append(column_name)
		elif pd.api.types.is_integer_dtype(series) or pd.api.types.is_categorical_dtype(series):
			# Consider low-cardinality numeric as categorical
			unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
			if unique_ratio < 0.05 and series.nunique(dropna=True) <= 50:
				categorical_cols.append(column_name)
	return categorical_cols


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
	if df.empty:
		return []
	return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
	if df.empty:
		return []
	return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def apply_filters(
	df: pd.DataFrame,
	date_col: Optional[str],
	date_range: Optional[List[pd.Timestamp]],
	categorical_selections: dict,
) -> pd.DataFrame:
	filtered = df.copy()
	if date_col and date_range is not None and len(date_range) == 2:
		start_date, end_date = date_range
		mask = filtered[date_col].between(start_date, end_date, inclusive="both")
		filtered = filtered.loc[mask]

	for col_name, selected_values in categorical_selections.items():
		if selected_values:
			filtered = filtered[filtered[col_name].isin(selected_values)]

	return filtered


def bytes_for_excel_download(df: pd.DataFrame) -> bytes:
	buffer = io.BytesIO()
	with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
		df.to_excel(writer, index=False, sheet_name="data")
	return buffer.getvalue()


def generate_report_md(
	df: pd.DataFrame,
	metric_col: Optional[str],
	group_col: Optional[str],
	date_col: Optional[str],
	file_name: Optional[str] = None,
) -> str:
	lines: List[str] = []
	lines.append("# Model Inference Dashboard Report")
	lines.append("")
	lines.append("## Data Source")
	if file_name:
		lines.append(f"File: {file_name}")
	else:
		lines.append("File: Default experiments.csv")
	lines.append("")
	lines.append("## Overview")
	lines.append(f"Total rows (after filters): {len(df):,}")
	if metric_col:
		lines.append(f"Metric column: {metric_col}")
	if group_col:
		lines.append(f"Category column: {group_col}")
	if date_col:
		lines.append(f"Date column: {date_col}")

	lines.append("")
	lines.append("## Summary Statistics")
	if metric_col and metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]):
		series = df[metric_col].dropna()
		if not series.empty:
			lines.append(f"Count: {series.count():,}")
			lines.append(f"Mean: {series.mean():.4f}")
			lines.append(f"Std: {series.std():.4f}")
			lines.append(f"Min: {series.min():.4f}")
			lines.append(f"Max: {series.max():.4f}")
		else:
			lines.append("No numeric values available for the selected metric.")
	else:
		lines.append("No metric selected or metric is not numeric.")

	return "\n".join(lines)


def main() -> None:
	st.title("📊 Model Inference Dashboard")
	
	# Sidebar for data import
	with st.sidebar:
		st.header("📁 Data Import")
		st.caption("Choose your data source")
		
		# Data source selection
		data_source = st.radio(
			"Select data source:",
			["📄 CSV File", "🗄️ PostgreSQL", "📊 Default Data"],
			index=2
		)
		
		df = pd.DataFrame()
		uploaded_file = None
		data_source_name = "Default CSV"
		table_name = None
		
		if data_source == "📄 CSV File":
			uploaded_file = st.file_uploader(
				"Choose a CSV file",
				type=['csv'],
				help="Upload your own CSV file to analyze."
			)
			
			if uploaded_file is not None:
				st.success(f"✅ File uploaded: {uploaded_file.name}")
				df = load_uploaded_file(uploaded_file)
				data_source_name = f"Uploaded CSV: {uploaded_file.name}"
				
				if df.empty:
					st.error("Failed to load uploaded file. Please check the file format.")
					st.stop()
				
				# Add clear button
				if st.button("🗑️ Clear uploaded file", type="secondary"):
					st.rerun()
			else:
				st.info("Please upload a CSV file to continue.")
				st.stop()
				
		elif data_source == "🗄️ PostgreSQL":
			if not SQLALCHEMY_AVAILABLE:
				st.error("SQLAlchemy support not available. Install sqlalchemy.")
				st.stop()
			
			if not CONFIG_AVAILABLE:
				st.error("Database configuration not available. Check config.py.")
				st.stop()
			
			# Query options
			query_option = st.radio(
				"Query type:",
				["📋 Simple Table", "🔍 Custom SQL Query"],
				index=0
			)
			
			if query_option == "📋 Simple Table":
				# Allow custom table name
				table_name = st.text_input(
					"Table name",
					value=POSTGRES_CONFIG.get('table_name', 'experiments'),
					help="Enter the table name to load data from"
				)
				
				if st.button("🔄 Load from PostgreSQL", type="primary"):
					with st.spinner("Connecting to PostgreSQL..."):
						df = load_from_postgres(table_name=table_name)
						data_source_name = f"PostgreSQL: {table_name}"
					
					if df.empty:
						st.error("Failed to load data from PostgreSQL. Check your connection settings.")
						st.stop()
					else:
						st.success(f"✅ Loaded {len(df):,} rows from PostgreSQL table: {table_name}")
			
			else:  # Custom SQL Query
				st.info(" Enter a custom SQL query to load data")
				custom_query = st.text_area(
					"SQL Query",
					value="SELECT * FROM experiments LIMIT 1000",
					height=100,
					help="Enter your SQL query here"
				)
				
				if st.button("🔄 Execute Query", type="primary"):
					with st.spinner("Executing SQL query..."):
						df = load_from_postgres(custom_query=custom_query)
						data_source_name = "PostgreSQL: Custom Query"
					
					if df.empty:
						st.error("Failed to execute query. Check your SQL syntax and connection settings.")
						st.stop()
					else:
						st.success(f"✅ Query executed successfully. Loaded {len(df):,} rows.")
			
			if df.empty:
				st.info("Click 'Load from PostgreSQL' or 'Execute Query' to fetch data.")
				st.stop()
				
		else:  # Default Data
			st.info("📊 Using default data from ./data/experiments.csv")
			csv_path = "./data/experiments.csv"
			df = load_experiments(csv_path)
			data_source_name = "Default experiments.csv"
			
			if df.empty:
				st.error("Default data file not found. Please upload a CSV file or connect to PostgreSQL.")
				st.stop()

	st.caption(f"Interactive analysis of model inference results - {len(df):,} rows loaded from {data_source_name}")

	# Sidebar controls
	with st.sidebar:
		st.divider()
		st.header("🔍 Filters")
		datetime_cols = get_datetime_columns(df)
		categorical_cols = get_categorical_columns(df)
		numeric_cols = get_numeric_columns(df)

		date_col = None
		date_range: Optional[List[pd.Timestamp]] = None
		if datetime_cols:
			date_col = st.selectbox("Date column", datetime_cols, index=0)
			col_min = pd.to_datetime(df[date_col].min())
			col_max = pd.to_datetime(df[date_col].max())
			default_start = max(col_min, col_max - pd.Timedelta(days=30)) if pd.notna(col_min) and pd.notna(col_max) else col_min
			date_range = st.date_input(
				"Date range",
				value=(default_start.date() if pd.notna(default_start) else None, col_max.date() if pd.notna(col_max) else None),
				min_value=col_min.date() if pd.notna(col_min) else None,
				max_value=col_max.date() if pd.notna(col_max) else None,
			)
			# Convert Python dates back to pandas timestamps
			if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
				date_range = [pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])]

		categorical_selections: dict = {}
		if categorical_cols:
			for col_name in categorical_cols:
				unique_values = sorted([v for v in df[col_name].dropna().unique().tolist()])
				selected = st.multiselect(f"{col_name}", options=unique_values, default=[])
				categorical_selections[col_name] = selected

		st.divider()
		st.header("📈 Data Overview")
		st.metric("Total Rows", f"{len(df):,}")
		st.metric("Total Columns", f"{len(df.columns):,}")
		
		if not df.empty:
			st.caption("Column types:")
			numeric_count = len(get_numeric_columns(df))
			categorical_count = len(get_categorical_columns(df))
			datetime_count = len(get_datetime_columns(df))
			st.write(f"• Numeric: {numeric_count}")
			st.write(f"• Categorical: {categorical_count}")
			st.write(f"• Datetime: {datetime_count}")
		
		st.divider()
		st.header("📊 Metrics")
		metric_col = st.selectbox("Select numeric metric", options=["(none)"] + numeric_cols, index=0)
		metric_col = None if metric_col == "(none)" else metric_col

		group_col = None
		if categorical_cols:
			group_col = st.selectbox("Group by (categorical)", options=["(none)"] + categorical_cols, index=0)
			group_col = None if group_col == "(none)" else group_col

	filtered_df = apply_filters(df, date_col, date_range, categorical_selections)

	# Data preview
	st.subheader("📋 Data Preview")
	st.dataframe(df.head(10), use_container_width=True)
	st.caption(f"Showing first 10 rows of {len(df):,} total rows")
	
	st.divider()
	
	# KPI metrics
	st.subheader("Key Metrics")
	kpi_cols = st.columns(4)
	kpi_cols[0].metric("Rows", f"{len(filtered_df):,}")
	if metric_col and metric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[metric_col]):
		series = filtered_df[metric_col].dropna()
		kpi_cols[1].metric("Mean", f"{series.mean():.4f}")
		kpi_cols[2].metric("Median", f"{series.median():.4f}")
		kpi_cols[3].metric("Std", f"{series.std():.4f}")
	else:
		kpi_cols[1].metric("Mean", "-")
		kpi_cols[2].metric("Median", "-")
		kpi_cols[3].metric("Std", "-")

	st.divider()

	# Charts
	st.subheader("Charts")
	chart_tabs = st.tabs(["Time Series", "Bar by Category", "Data View"])

	with chart_tabs[0]:
		if date_col and metric_col and pd.api.types.is_numeric_dtype(filtered_df.get(metric_col, pd.Series(dtype=float))):
			agg_df = (
				filtered_df.dropna(subset=[date_col, metric_col])
				.groupby(pd.Grouper(key=date_col, freq="D"))
				[metric_col]
				.mean()
				.reset_index()
			)
			if not agg_df.empty:
				line = (
					alt.Chart(agg_df)
					.mark_line(point=True)
					.encode(x=alt.X(f"{date_col}:T", title="Date"), y=alt.Y(f"{metric_col}:Q", title="Mean"))
					.properties(height=320)
				)
				st.altair_chart(line, use_container_width=True)
			else:
				st.info("Not enough data to draw the time series.")
		else:
			st.info("Select a date column and a numeric metric to view the time series.")

	with chart_tabs[1]:
		if group_col and metric_col and pd.api.types.is_numeric_dtype(filtered_df.get(metric_col, pd.Series(dtype=float))):
			agg_df = (
				filtered_df.dropna(subset=[group_col, metric_col])
				.groupby(group_col)[metric_col]
				.mean()
				.reset_index()
			)
			if not agg_df.empty:
				bar = (
					alt.Chart(agg_df)
					.mark_bar()
					.encode(x=alt.X(group_col, sort="-y", title=group_col), y=alt.Y(metric_col, title="Mean"))
					.properties(height=320)
				)
				st.altair_chart(bar, use_container_width=True)
			else:
				st.info("Not enough data to draw the bar chart.")
		else:
			st.info("Select a categorical group and a numeric metric to view the bar chart.")

	with chart_tabs[2]:
		st.dataframe(filtered_df, use_container_width=True)

	st.divider()

	# Export section
	st.subheader("Export")
	col_a, col_b, col_c = st.columns(3)

	# Determine base filename for exports
	if uploaded_file is not None:
		base_name = uploaded_file.name.replace('.csv', '')
	else:
		base_name = "experiments"

	csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
	col_a.download_button(
		"Download CSV",
		data=csv_bytes,
		file_name=f"{base_name}_filtered.csv",
		mime="text/csv",
		use_container_width=True,
	)

	excel_bytes = bytes_for_excel_download(filtered_df)
	col_b.download_button(
		"Download Excel",
		data=excel_bytes,
		file_name=f"{base_name}_filtered.xlsx",
		mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		use_container_width=True,
	)

	report_md = generate_report_md(
		filtered_df,
		metric_col=metric_col,
		group_col=group_col,
		date_col=date_col,
		file_name=uploaded_file.name if uploaded_file is not None else None,
	)
	col_c.download_button(
		"Download Report (.md)",
		data=report_md.encode("utf-8"),
		file_name="inference_report.md",
		mime="text/markdown",
		use_container_width=True,
	)


if __name__ == "__main__":
	main()



