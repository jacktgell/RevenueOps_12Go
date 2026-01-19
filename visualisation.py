import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

class DataVisualization():
    """
    Standardized visualization tools for the revenue pipeline.
    """

    def run(self, df: pd.DataFrame):
        self.log("Generating summary plots...")
        self.plot_visit_distribution(df)

    def plot_visit_distribution(self, df: pd.DataFrame):
        """Plots the histogram of sessions over time."""
        plt.figure(figsize=(12, 6))

        # We drop NaT values to avoid plotting errors
        valid_dates = df['visit_date'].dropna()

        # Plot histogram
        plt.hist(valid_dates, bins=30, color='#2c3e50', edgecolor='white', alpha=0.9)

        # Formatting
        plt.title(f'Distribution of Visitor Sessions (N={len(valid_dates)})', fontsize=14)
        plt.xlabel('Visit Date', fontsize=12)
        plt.ylabel('Frequency (Sessions)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Highlight: This proves we fixed the "mixed date format" issue
        plt.tight_layout()
        plt.show()

        self.log("Plot generated. Verify that there are no outliers in 1970 or 2100.")

    @staticmethod
    def plot_precision_funnel(df):
        print("--- Generating Funnel (2 Decimal Places) ---")
        print(f'Vistors: {len(df)}')
        print(f'Clicked trip: {df['clicked_trip'].sum()}')
        print(f'Added to cart: {df['added_to_cart'].sum()}')
        print(f'Bookings: {df['booked'].sum()}')

        # 1. Calculate Data
        visitors = len(df)
        clicks = df['clicked_trip'].sum()
        atc = df['added_to_cart'].sum()
        bookings = df['booked'].sum()

        values = [visitors, clicks, atc, bookings]
        stages = ['Total Visitors', 'Clicked Trip', 'Added to Cart', 'Booked']

        # 2. Define Positions & Colors
        positions = ["inside", "outside", "outside", "outside"]
        text_colors = ["white", "black", "black", "black"]

        # 3. Create the Funnel
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,

            # KEY CHANGE: Using texttemplate for custom formatting
            # %{value:,d}    -> Adds comma separators (e.g. 1,000)
            # %{percentInitial:.2%} -> 2 decimal places (e.g. 12.34%)
            texttemplate="%{value:,d}<br>%{percentInitial:.2%} of Total<br>%{percentPrevious:.2%} of Prev",

            textposition=positions,
            textfont=dict(color=text_colors, size=13),

            marker=dict(
                color=["#5DADE2", "#F5B041", "#E59866", "#58D68D"],
                line=dict(width=0)
            ),
            connector={"line": {"color": "white", "width": 0}},
            opacity=0.85
        ))

        # 4. Styling
        fig.update_layout(
            title_text="<b>12Go Conversion Funnel</b>",
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="white",
            height=500,
            width=950,
            showlegend=False,
            margin=dict(r=200, l=100)
        )

        fig.show()

    def plot_comparative_funnel(self, df: pd.DataFrame):
        """
        Plots two funnels side-by-side: Control (0) vs Dynamic Pricing (1).
        Prints detailed counts for both groups.
        """
        print("--- Generating Comparative Funnels ---")

        # 1. Split Data
        df_control = df[df['experiment_variation_id'] == 0]
        df_dynamic = df[df['experiment_variation_id'] == 1]

        # 2. Print Metrics (The requested addition)
        print("--- Control Group Metrics (Variation 0) ---")
        print(f"Visitors: {len(df_control)}")
        print(f"Clicked trip: {df_control['clicked_trip'].sum()}")
        print(f"Added to cart: {df_control['added_to_cart'].sum()}")
        print(f"Bookings: {df_control['booked'].sum()}")

        print("--- Dynamic Group Metrics (Variation 1) ---")
        print(f"Visitors: {len(df_dynamic)}")
        print(f"Clicked trip: {df_dynamic['clicked_trip'].sum()}")
        print(f"Added to cart: {df_dynamic['added_to_cart'].sum()}")
        print(f"Bookings: {df_dynamic['booked'].sum()}")

        # 3. Prepare Data for Plotting
        stages = ['Total Visitors', 'Clicked Trip', 'Added to Cart', 'Booked']

        values_control = [
            len(df_control),
            df_control['clicked_trip'].sum(),
            df_control['added_to_cart'].sum(),
            df_control['booked'].sum()
        ]

        values_dynamic = [
            len(df_dynamic),
            df_dynamic['clicked_trip'].sum(),
            df_dynamic['added_to_cart'].sum(),
            df_dynamic['booked'].sum()
        ]

        # 4. Create Subplots (1 Row, 2 Cols)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("<b>Control Group</b> (Baseline)", "<b>Experiment</b> (Dynamic)"),
            shared_yaxes=True
        )

        # 5. Define Styling
        positions = ["inside", "outside", "outside", "outside"]
        text_colors = ["white", "black", "black", "black"]
        colors = ["#5DADE2", "#F5B041", "#E59866", "#58D68D"]

        # Template: Value + % of Initial + % of Previous
        template = "%{value:,d}<br>%{percentInitial:.1%} of Total<br>%{percentPrevious:.1%} of Prev"

        # 6. Add Traces
        # -- Control Group --
        fig.add_trace(go.Funnel(
            name='Control',
            y=stages,
            x=values_control,
            textposition=positions,
            texttemplate=template,
            textfont=dict(color=text_colors, size=12),
            marker=dict(color=colors, line=dict(width=0)),
            connector={"line": {"color": "white", "width": 0}},
            opacity=0.85
        ), row=1, col=1)

        # -- Dynamic Group --
        fig.add_trace(go.Funnel(
            name='Dynamic',
            y=stages,
            x=values_dynamic,
            textposition=positions,
            texttemplate=template,
            textfont=dict(color=text_colors, size=12),
            marker=dict(color=colors, line=dict(width=0)),
            connector={"line": {"color": "white", "width": 0}},
            opacity=0.85
        ), row=1, col=2)

        # 7. Final Layout
        fig.update_layout(
            title_text="<b>Funnel Comparison: Control vs. Dynamic Pricing</b>",
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="white",
            height=500,
            width=1100,
            showlegend=False,
            margin=dict(r=50, l=100)
        )

        fig.show()

    def analyze_financial_impact(self, df):
        print("--- Financial Impact: Control vs Dynamic ---")

        # 1. Aggregate Core Metrics
        impact = df.groupby('experiment_variation_id').agg(
            Visitors=('visitor_id', 'size'),
            Bookings=('booked', 'sum'),
            Total_Revenue=('revenue', 'sum'),
            Total_Profit=('netprofit', 'sum')
        )

        # 2. Calculate Derived Efficiency Metrics
        impact['RPV'] = impact['Total_Revenue'] / impact['Visitors']
        impact['AOV'] = impact['Total_Revenue'] / impact['Bookings']
        impact['Profit_per_Visitor'] = impact['Total_Profit'] / impact['Visitors']

        # 3. Transpose for "Side-by-Side" view
        impact = impact.T
        impact.columns = ['Control (0)', 'Dynamic (1)']

        # 4. Calculate Lift
        impact['Lift (%)'] = ((impact['Dynamic (1)'] - impact['Control (0)']) / impact['Control (0)']) * 100

        # 5. formatting style (Optional, for clean notebook output)
        # We return the raw dataframe so you can see it easily
        return impact

    def check_sample_balance(self, df):
        print("--- HYPOTHESIS TESTING: Sample Balance ---")

        # 1. Calculate % of Transport Type for Control Group (0)
        control_mix = df[df['experiment_variation_id'] == 0]['transport_type'].value_counts(normalize=True)

        # 2. Calculate % of Transport Type for Dynamic Group (1)
        dynamic_mix = df[df['experiment_variation_id'] == 1]['transport_type'].value_counts(normalize=True)

        # 3. Combine into a clean DataFrame
        balance_df = pd.DataFrame({
            'Control %': control_mix,
            'Dynamic %': dynamic_mix
        })

        # 4. Calculate Difference to spot issues
        balance_df['Diff'] = balance_df['Dynamic %'] - balance_df['Control %']

        # 5. Display with readable formatting
        #    (Background gradient helps visualize the biggest categories)
        print("Transport Type Distribution:")
        return balance_df.style.format("{:.2%}").background_gradient(cmap='Blues', subset=['Control %', 'Dynamic %'])

    def analyze_segment_performance(self, df, segment_col='transport_type'):
        """
        Analyzes funnel performance by a specific segment (Transport or Route).
        Shows ALL segments (sorted by volume).
        """

        # 1. Handle Special 'Route' Logic
        work_df = df.copy()
        if segment_col == 'route':
            work_df['route'] = work_df['origin'] + " -> " + work_df['destination']
            groupby_col = 'route'
            title_name = "Route"
        else:
            groupby_col = segment_col
            title_name = segment_col.replace('_', ' ').title()

        print(f"--- Funnel Performance by {title_name} (All Segments) ---")

        # 2. Aggregation
        stats = work_df.groupby(groupby_col).agg(
            Visitors=('visitor_id', 'count'),
            Clicks=('clicked_trip', 'sum'),
            ATCs=('added_to_cart', 'sum'),
            Bookings=('booked', 'sum')
        )

        # 3. Calculate Rates (%)
        stats['CTR'] = (stats['Clicks'] / stats['Visitors']) * 100
        stats['ATC_Rate'] = (stats['ATCs'] / stats['Visitors']) * 100
        stats['CVR'] = (stats['Bookings'] / stats['Visitors']) * 100

        # 4. Sort by Volume (Highest to Lowest)
        stats = stats.sort_values('Visitors', ascending=False)

        # 5. Create the Grouped Bar Chart
        fig = go.Figure()
        categories = stats.index

        # Metric 1: CTR (Interest)
        fig.add_trace(go.Bar(
            x=categories, y=stats['CTR'],
            name='CTR (Click %)',
            marker_color='#5DADE2',  # Light Blue
            text=stats['CTR'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto'
        ))

        # Metric 2: ATC Rate (Intent)
        fig.add_trace(go.Bar(
            x=categories, y=stats['ATC_Rate'],
            name='ATC Rate (Cart %)',
            marker_color='#F5B041',  # Orange
            text=stats['ATC_Rate'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto'
        ))

        # Metric 3: CVR (Conversion)
        fig.add_trace(go.Bar(
            x=categories, y=stats['CVR'],
            name='CVR (Booking %)',
            marker_color='#58D68D',  # Green
            text=stats['CVR'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto'
        ))

        fig.update_layout(
            title=f"<b>Funnel Metrics by {title_name}</b>",
            title_x=0.5,
            yaxis_title="Rate (%)",
            barmode='group',
            plot_bgcolor='white',
            height=600,
            xaxis_tickangle=-45
        )

        fig.show()

        # 6. Print the Data Table
        print(f"--- Detailed Data Table: {title_name} ---")

        format_dict = {
            'Visitors': "{:,}",
            'Clicks': "{:,}",
            'ATCs': "{:,}",
            'Bookings': "{:,}",
            'CTR': "{:.2f}%",
            'ATC_Rate': "{:.2f}%",
            'CVR': "{:.2f}%"
        }

        display(stats.style.format(format_dict).background_gradient(cmap='Blues', subset=['CTR', 'ATC_Rate', 'CVR']))

    def check_demographic_balance(self, df):
        print("--- HYPOTHESIS TESTING: User Demographics Balance ---")

        # List of categorical columns to check
        demographics = ['device', 'os', 'utm_source', 'country']

        for col in demographics:
            print(f"Checking: {col.upper()}")

            # 1. Create a Crosstab (Frequency Table)
            balance = pd.crosstab(
                df[col],
                df['experiment_variation_id'],
                normalize='columns'
            ) * 100

            # 2. Rename columns
            balance.columns = ['Control %', 'Dynamic %']

            # 3. Calculate Difference
            balance['Diff'] = balance['Dynamic %'] - balance['Control %']

            # 4. Sort and Limit
            balance = balance.sort_values('Control %', ascending=False).head(10)

            # 5. Display with 'coolwarm' cmap (Blue <-> Red)
            # vmin/vmax ensures 0 is the center (white/neutral)
            display(balance.style.format("{:.2f}%").background_gradient(
                subset=['Diff'], cmap='coolwarm', vmin=-2, vmax=2
            ))

    def analyze_daily_booking_curve(self, df):
        print("--- Daily Booking Curve (Dynamic Range) ---")

        # 1. Prepare Data
        df_clean = df.copy()

        # Ensure datetime
        df_clean['visit_date'] = pd.to_datetime(df_clean['visit_date'])
        df_clean['trip_date'] = pd.to_datetime(df_clean['trip_date'])

        # Calculate Lead Time (Days)
        df_clean['days_to_trip'] = (df_clean['trip_date'] - df_clean['visit_date']).dt.days

        # Filter: Valid Lead Times (>= 0) and Booked users only
        # Cap at 99th percentile to remove extreme outliers
        cap = df_clean['days_to_trip'].quantile(0.99)
        mask = (df_clean['days_to_trip'] >= 0) & (df_clean['days_to_trip'] <= cap) & (df_clean['booked'] == 1)
        df_daily = df_clean[mask]

        # 2. Aggregation by Day
        stats = df_daily.groupby('days_to_trip').agg(
            Booking_Volume=('booked', 'count'),
            Total_Revenue=('price_paid', 'sum'),
            Total_Net_Profit=('netprofit', 'sum')
        )

        # 3. Fill Missing Days
        max_day = int(stats.index.max())
        all_days = pd.Index(range(max_day + 1), name='days_to_trip')
        stats = stats.reindex(all_days, fill_value=0)

        # 4. Calculate Averages
        stats['AOV'] = stats['Total_Revenue'] / stats['Booking_Volume']
        stats['Avg_Net_Profit'] = stats['Total_Net_Profit'] / stats['Booking_Volume']

        # Fill NaN with 0
        stats[['AOV', 'Avg_Net_Profit']] = stats[['AOV', 'Avg_Net_Profit']].fillna(0)

        # 5. Visualisation
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar: Volume (Left Axis)
        fig.add_trace(
            go.Bar(
                x=stats.index,
                y=stats['Booking_Volume'],
                name="Booking Volume (#)",
                marker_color='#5DADE2',  # Blue
                opacity=0.4
            ),
            secondary_y=False
        )

        # Line 1: AOV (Right Axis)
        fig.add_trace(
            go.Scatter(
                x=stats.index,
                y=stats['AOV'],
                name="Avg Order Value ($)",
                marker_color='#E74C3C',  # Red
                mode='lines',
                line=dict(width=2)
            ),
            secondary_y=True
        )

        # Line 2: Avg Net Profit (Right Axis)
        fig.add_trace(
            go.Scatter(
                x=stats.index,
                y=stats['Avg_Net_Profit'],
                name="Avg Net Profit ($)",
                marker_color='#2ECC71',  # Green
                mode='lines',
                line=dict(width=3)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=f"<b>Daily Demand Curve (0 to {max_day} Days)</b>",
            title_x=0.5,
            plot_bgcolor='white',
            height=600,
            xaxis_title="Days Before Trip",
            legend=dict(x=0.5, y=1.1, orientation='h')
        )

        fig.update_yaxes(title_text="Booking Volume (#)", secondary_y=False)
        fig.update_yaxes(title_text="Value & Profit ($)", secondary_y=True)

        fig.show()

        # 6. Display Dense Table
        print("--- Detailed Daily Metrics ---")

        format_dict = {
            'Booking_Volume': "{:,}",
            'Total_Revenue': "${:,.0f}",
            'Total_Net_Profit': "${:,.0f}",
            'AOV': "${:.2f}",
            'Avg_Net_Profit': "${:.2f}"
        }

        # Display table with gradient to spot trends (Greens for profit, Blues for volume)
        display(stats.style.format(format_dict)
                .background_gradient(subset=['Booking_Volume'], cmap='Blues')
                .background_gradient(subset=['Avg_Net_Profit'], cmap='Greens'))

    def plot_price_distributions_by_trip(self, df, n_cols=3):
        """
        Generates small multiple histograms for ALL routes.
        Includes fix for MultiIndex column flattening (int vs str).
        """
        print(f"--- Price Distribution by Trip (Grid View) ---")

        # 1. Identify All Trips (Sorted by Volume)
        trip_counts = df['trip_id'].value_counts()
        all_trips = trip_counts.index.tolist()
        n_trips = len(all_trips)

        # 2. Calculate Rows needed
        n_rows = (n_trips + n_cols - 1) // n_cols

        # Dynamic Spacing
        v_space = 0.3 / n_rows
        v_space = max(0.02, min(0.15, v_space))

        print(f"Plotting {n_trips} trips in {n_rows} rows x {n_cols} cols (Spacing: {v_space:.3f})...")

        # 3. Setup Subplot Grid
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"ID: {tid}" for tid in all_trips],
            vertical_spacing=v_space,
            horizontal_spacing=0.05
        )

        # 4. Loop through All Trips
        for i, tid in enumerate(all_trips):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1

            trip_data = df[df['trip_id'] == tid]

            origin = trip_data['origin'].iloc[0]
            dest = trip_data['destination'].iloc[0]
            transport = trip_data['transport_type'].iloc[0]
            label = f"{origin} -> {dest}<br>({transport})"

            # Update Subplot Title
            fig.layout.annotations[i].text = f"<b>{label}</b>"
            fig.layout.annotations[i].font.size = 10

            # Control Group
            fig.add_trace(go.Histogram(
                x=trip_data[trip_data['experiment_variation_id'] == 0]['price_shown'],
                name='Control',
                marker_color='#34495E',
                opacity=0.75,
                nbinsx=15,
                showlegend=(i == 0)
            ), row=row, col=col)

            # Dynamic Group
            fig.add_trace(go.Histogram(
                x=trip_data[trip_data['experiment_variation_id'] == 1]['price_shown'],
                name='Dynamic',
                marker_color='#E74C3C',
                opacity=0.6,
                nbinsx=15,
                showlegend=(i == 0)
            ), row=row, col=col)

        # 5. Dynamic Layout
        fig.update_layout(
            title=f"<b>Price Strategy per Route (All {n_trips} Trips)</b>",
            height=300 * n_rows,
            width=1100,
            plot_bgcolor='white',
            barmode='overlay',
            margin=dict(t=100, b=50)
        )

        fig.show()

        # --- DENSE TABLE ---
        print(f"--- Detailed Statistics: All Trips ---")
        stats_df = df.copy()
        stats_df['route_name'] = stats_df['origin'] + "->" + stats_df['destination'] + " (" + stats_df[
            'transport_type'] + ")"

        # Calculate Stats
        summary = stats_df.groupby(['route_name', 'experiment_variation_id'])['price_shown'].describe()[
            ['mean', 'std', 'min', 'max']]
        summary = summary.unstack()

        # --- THE FIX IS HERE ---
        # We use map(str, col) to convert (mean, 0) -> ('mean', '0') so .join works
        summary.columns = ['_'.join(map(str, col)).strip() for col in summary.columns.values]

        # Calculate Differences
        summary['Mean_Diff'] = summary['mean_1'] - summary['mean_0']
        summary['Std_Diff'] = summary['std_1'] - summary['std_0']

        # Add Volume
        volume = stats_df.groupby('route_name')['visitor_id'].count()
        summary['Volume'] = volume

        # Clean up
        final_cols = ['Volume', 'mean_0', 'mean_1', 'Mean_Diff', 'std_0', 'std_1', 'Std_Diff', 'min_1', 'max_1']
        summary = summary[final_cols]
        summary.columns = ['Vol', 'Avg(Ctrl)', 'Avg(Dyn)', 'Avg Diff', 'Std(Ctrl)', 'Std(Dyn)', 'Std Diff', 'Min(Dyn)',
                           'Max(Dyn)']
        summary = summary.sort_values('Vol', ascending=False)

        display(summary.style.format({
            'Vol': "{:,}",
            'Avg(Ctrl)': "{:,.2f}", 'Avg(Dyn)': "{:,.2f}", 'Avg Diff': "{:,.2f}",
            'Std(Ctrl)': "{:,.2f}", 'Std(Dyn)': "{:,.2f}", 'Std Diff': "{:,.2f}",
            'Min(Dyn)': "{:,.2f}", 'Max(Dyn)': "{:,.2f}"
        }).background_gradient(subset=['Avg Diff'], cmap='coolwarm')
                .background_gradient(subset=['Std Diff'], cmap='Greens'))

    def plot_day_of_week_seasonality(self, df):
        """
        Checks if specific days of the week drive higher conversion.
        """
        print("--- Analyzing Weekly Seasonality ---")

        # 1. Prepare Data
        df = df.copy()
        df['dow'] = pd.to_datetime(df['visit_date']).dt.day_name()

        # Order of days (Monday -> Sunday)
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # 2. Aggregation
        stats = df.groupby('dow').agg(
            Visitors=('visitor_id', 'count'),
            Bookings=('booked', 'sum'),
            Revenue=('price_paid', 'sum')
        ).reindex(order)

        # 3. Calculated Metrics
        stats['CVR'] = (stats['Bookings'] / stats['Visitors']) * 100
        stats['AOV'] = stats.apply(lambda x: x['Revenue'] / x['Bookings'] if x['Bookings'] > 0 else 0, axis=1)

        # 4. Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar: Visitors
        fig.add_trace(
            go.Bar(x=stats.index, y=stats['Visitors'], name='Visitors', marker_color='#BDC3C7'),
            secondary_y=False
        )

        # Line: CVR
        fig.add_trace(
            go.Scatter(x=stats.index, y=stats['CVR'], name='Conversion Rate %',
                       line=dict(color='#2980B9', width=3)),
            secondary_y=True
        )

        fig.update_layout(
            title="<b>Weekly Seasonality: Traffic vs Conversion</b>",
            title_x=0.5,
            plot_bgcolor='white',
            height=500
        )

        fig.show()

        # --- TABLE: Detailed Daily Metrics ---
        print("--- Detailed Weekly Metrics ---")
        format_dict = {
            'Visitors': "{:,}",
            'Bookings': "{:,}",
            'Revenue': "${:,.0f}",
            'CVR': "{:.2f}%",
            'AOV': "${:.2f}"
        }
        display(stats.style.format(format_dict).background_gradient(subset=['CVR', 'AOV'], cmap='Greens'))

    def plot_correlation_matrix(self, df):
        """
        Checks for multicollinearity between numerical features.
        """

        # Select numerical columns relevant for modeling
        cols = ['price_shown', 'distance_km', 'duration_min', 'seats_left',
                'days_to_trip', 'popularity_score', 'booked']

        # Calculate days_to_trip on the fly if needed
        work_df = df.copy()
        if 'days_to_trip' not in work_df.columns:
            work_df['trip_date'] = pd.to_datetime(work_df['trip_date'])
            work_df['visit_date'] = pd.to_datetime(work_df['visit_date'])
            work_df['days_to_trip'] = (work_df['trip_date'] - work_df['visit_date']).dt.days

        # Compute Correlation
        corr = work_df[cols].corr()

        # --- TABLE: Correlation Data ---
        print("--- Correlation Coefficients ---")
        display(corr.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1).format("{:.2f}"))
