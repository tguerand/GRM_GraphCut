import pandas as pd
from shapely import wkt

pd.set_option('max_colwidth', 1000)
W = 3349
H = 3391


def scale_x_and_y_coords_to_pixels(x: float, y: float, x_max: float, y_min: float):
    w_prime = W*W/(W+1)
    h_prime = H*H/(H+1)

    x_prime = x/x_max*w_prime
    y_prime = y/y_min*h_prime

    return int(x_prime), int(y_prime)


def apply_shapely_on_geometries(geometry: str):
    geometry_wkt = wkt.loads(geometry)
    return list(geometry_wkt)


def extract_pixels_from_polygons(row):
    polygon_wkt = row['geom']
    x_max = row['Xmax']
    y_min = row['Ymin']

    coords = list(polygon_wkt.exterior.coords)
    list_pixels = [scale_x_and_y_coords_to_pixels(*coord, x_max, y_min) for coord in coords]
    return list_pixels


class Loader:
    """
    Create a dataframe with one polygon per line that are rescaled to the pixels format
    """
    def __init__(self, wkt_csv, grid_size_csv):
        df_wkt = pd.read_csv(wkt_csv, dtype={'ImageId': str})
        df_grid_size = pd.read_csv(grid_size_csv, dtype={'ImageId': str})
        self.df = df_wkt.merge(df_grid_size, on=['ImageId'])

    def preprocess(self):
        self.df['geom'] = self.df['MultipolygonWKT'].map(apply_shapely_on_geometries)
        self.df = self.df.explode('geom', ignore_index=True)
        self.df = self.df.dropna().drop(['MultipolygonWKT'], axis=1)

        self.df['geom'] = self.df.apply(extract_pixels_from_polygons, axis=1)

    def save_final_df(self, out_path='df_with_polygons_as_pixels.csv'):
        self.preprocess()
        self.df[['ImageId', 'geom', 'Xmax', 'Ymin']].to_csv(out_path, index=False)


if __name__ == '__main__':
    Loader('train_wkt_dataset.csv', 'grid_sizes_dataset.csv').save_final_df()
