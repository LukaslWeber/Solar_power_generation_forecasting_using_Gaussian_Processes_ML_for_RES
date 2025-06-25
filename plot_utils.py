import contextily as ctx
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
from shapely.geometry import Point


def plot_weather_correlation(df):
    df_numerical = df.drop(columns=['time'])

    corr_matrix = df_numerical.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                cbar_kws={"shrink": .8}, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_solar_supply_day(random_date, supply_hourly):
    mask = supply_hourly['time'].dt.date == random_date
    df_day = supply_hourly[mask].sort_values('time')

    plt.figure(figsize=(8, 3))
    plt.plot(df_day['time'], df_day['solar_supply_sum'], color='green', linewidth=2)
    plt.title(f'Solar Supply on {random_date.strftime("%-d.%-m.%Y")}', fontsize=12)
    plt.xlabel('Time (Hourly)', fontsize=10)
    plt.ylabel('Solar Supply (Sum per hour)', fontsize=10)

    # Format x-axis for hours
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Grid and layout
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_solar_supply_whole(supply_hourly):
    df_sorted = supply_hourly.sort_values('time', ascending=True)

    # Create the plot
    plt.figure(figsize=(11, 6))
    plt.plot(df_sorted['time'], df_sorted['solar_supply_sum'], color='green', linewidth=0.5)

    # Set axis labels
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Solar Supply (Sum per hour)', fontsize=12)
    plt.title('Hourly Solar Supply Over Time', fontsize=14)

    # Format x-axis ticks
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))

    plt.grid(True, which='major', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_installed_capacity(capacity_df):
    plt.figure()
    plt.plot(capacity_df.index, capacity_df.values)
    plt.xlabel('Time')
    plt.ylabel('Photovoltaic Capacity (MW)')
    plt.title('Hourly Interpolated Installed Photovoltaic Capacity')
    plt.xticks(rotation=-45)
    plt.tight_layout()
    plt.show()


def plot_time_columns(full_data):
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.5)

    # Plot of yearly time encoding
    plt.subplot(2, 1, 1)
    plt.plot(full_data.index, full_data['day_of_year_sin'], label="sin(day of year)")
    plt.plot(full_data.index, full_data['day_of_year_cos'], label="cos(day of year)")

    plt.title('Yearly Time Encoding')
    plt.xlabel('Date')
    plt.legend(loc='upper right')

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    # plt.xticks(rotation=-45)

    # Plot of daily time encoding
    plt.subplot(2, 1, 2)
    plt.plot(full_data.index[0:24 * 3], full_data['hour_sin'][0:24 * 3], label="sin(hour of day)")
    plt.plot(full_data.index[0:24 * 3], full_data['hour_cos'][0:24 * 3], label="cos(hour of day)")

    plt.title('Daily Time Encoding')
    plt.xlabel('Date')
    plt.legend(loc='upper right')

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%d.%m.%Y %H:00'))
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    # plt.xticks(rotation=0)

    plt.show()


def plot_forecast_sanity_check(i, X_train, y_train, horizon):
    plt.plot(X_train[i:i + horizon, 0].cpu().numpy(), label=f"X_train[{i}:{i}+24]")
    plt.plot(y_train[i].cpu().numpy(), label=f"y_train[{i}]")
    plt.legend()
    plt.show()


def plot_geographical_clusters(unique_locations):
    # Convert DataFrame to GeoDataFrame
    df = unique_locations.copy()
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # Convert to Web Mercator for plotting with contextily
    gdf = gdf.to_crs(epsg=3857)

    # choose 4 colors that correspond to the clusters
    colors = ['#457b9d', '#a8dadc', '#f4a261', '#e76f51']  # Cool blue/green and soft orange/red
    cmap = ListedColormap(colors)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, column='cluster', cmap=cmap, legend=False, markersize=30, alpha=0.9)

    # Add basemap
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Add legend
    legend_elements = [Patch(facecolor=colors[i], label=f'Cluster {i}') for i in range(4)]
    ax.legend(handles=legend_elements, title="Clusters", loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4,
              frameon=False)

    # Final touches
    ax.set_title("Clustered Geographical Points in and Around Germany")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
