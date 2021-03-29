import pandas as pd
from shapely import wkt

pd.set_option('max_colwidth', 1000)
W = 3349
H = 3391


def scale_x_and_y_coords_to_pixels(x: float, y: float, x_max: float, y_min: float):
    h_prime = W*W/(W+1)
    w_prime = H*H/(H+1)

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
        """
        ClassTypes:
            1 - Buildings - large building, residential, non-residential, fuel storage facility, fortified building
            2 - Misc. Manmade structures 
            3 - Road 
            4 - Track - poor/dirt/cart track, footpath/trail
            5 - Trees - woodland, hedgerows, groups of trees, standalone trees
            6 - Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
            7 - Waterway 
            8 - Standing water
            9 - Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
            10 - Vehicle Small - small vehicle (car, van), motorbike

        We will keep only the buildings, the roads and the forest
            it is the classes: 1, 3, 4 and 5

        """
        
        
        
        self.preprocess()
        
        #classes_to_keep = [1, 3, 4, 5]
        classes_to_keep = [4,3, 7, 8]
        classes = range(1,11)
        
        for cl in classes:
            if cl in classes_to_keep:
                continue
            idx = self.df.index[self.df['ClassType']==cl].tolist()
            self.df = self.df.drop(idx)
        self.df['geom_red'] = self.df.index
        self.df[['ImageId', 'geom', 'ClassType', 'Xmax', 'Ymin', 'geom_red']].to_csv(out_path, index=False)


if __name__ == '__main__':
    Loader(r'../df/train_wkt_dataset.csv', r'../df/grid_sizes_dataset.csv').save_final_df()
