import os
import numpy as np
import random
import uuid 
import PIL
import click
from PIL import ImageDraw, ImageFont
from iss.init_config import CONFIG
from iss.tools import Tools


##
##
##

##
## Variables globales
##


BORDER_SIZE = 30
O_WIDTH = 1080 
O_HEIGHT = 720
CLUSTERS = {
    'horizon_only': [2, 6, 20, 23, 27, 44, 47],
    'horizon_iss': [4, 5, 12, 17, 32, 37, 40],
    'vertical': [1, 22, 34, 38, 45]
}

COLORS = {
    'x': {'hex': '#f7f9f9', 'rgb': [247, 249, 249]},
    '2x': {'hex': '#bed8d4', 'rgb': [190, 216,212]},
    'x.2': {'hex': '#78d5d7', 'rgb': [120, 213, 215]},
    'x.rect': {'hex': '#63d2ff', 'rgb': [99, 210, 255]},
    '2x.bis': {'hex': '#2081c3', 'rgb': [32, 129, 195]}
}

POSTERS_CONFIG = [{
    "version": 1,
    "size":  {
        'width': O_WIDTH*4 + BORDER_SIZE*3, 
        'height': O_HEIGHT*4 + BORDER_SIZE*3,
        'channel': 3
    },
    "ARGS_SQL": (1, 'simple_conv', 'model_colab'),
    "DECOUPAGE": [
        {'origin': [0+BORDER_SIZE, 0+BORDER_SIZE], 'size': [1080*2, 720*2], 'cluster': [6, 23, 27, 44, 47], 'color': '2x', 'position': '1'},
        {'origin': [1080*2+BORDER_SIZE*2, 0+BORDER_SIZE], 'size': [1080-30, 720-30], 'cluster': [4, 12], 'color': 'x', 'position': '2'}, 
        {'origin': [1080*3+BORDER_SIZE*3-30, 0+BORDER_SIZE], 'size': [1080-30, 720-30], 'cluster': [32, 5, 17], 'color': 'x', 'position': '3'},
        {'origin': [1080*2+BORDER_SIZE*2, 720-30+2*BORDER_SIZE], 'size': [1080-30, 720-30], 'cluster': [1, 22, 34, 38, 45], 'color': 'x', 'position': '4'}, 
        {'origin': [1080*3+BORDER_SIZE*3-30, 720-30+2*BORDER_SIZE], 'size': [1080-30, 720-30], 'cluster': [6, 23, 27, 44, 47], 'color': 'x', 'position': '5'},

        {'origin': [0+BORDER_SIZE, 720*2+BORDER_SIZE*2], 'size': [1080-30, 720-30], 'cluster': [39], 'color': 'x', 'position': '6'}, 
        {'origin': [1080+BORDER_SIZE*2-30, 720*2+BORDER_SIZE*2], 'size': [1080-30, 720-30], 'cluster': [7], 'color': 'x', 'position': '7'}, 
        {'origin': [0+BORDER_SIZE, 720*3-30+BORDER_SIZE*3], 'size': [1080-30, 720-30], 'cluster': [1, 22, 34, 38, 45], 'color': 'x', 'position': '8'},
        {'origin': [1080+BORDER_SIZE*2-30, 720*3-30+BORDER_SIZE*3], 'size': [1080-30, 720-30], 'cluster': [46], 'color': 'x', 'position': '9'},
        {'origin': [1080*2+BORDER_SIZE*3-30*2, 720*2-30*2+BORDER_SIZE*3], 'size': [1080*2, 720*2], 'cluster': [6, 23, 27, 44, 47], 'color': '2x', 'position': '10'}
    ]
},
{
    "version": 2,
    "size":  {
        'width': O_WIDTH*4 + BORDER_SIZE*3, 
        'height': O_HEIGHT*4 + BORDER_SIZE*3,
        'channel': 3
    },
    "ARGS_SQL": (1, 'simple_conv', 'model_colab'),
    "DECOUPAGE": [
        {'origin': [0+BORDER_SIZE, 0+BORDER_SIZE], 'size': [1080*2, 720*2], 'cluster': CLUSTERS['horizon_only'][:4], 'color': '2x', 'position': '1'},
        {'origin': [1080*2+BORDER_SIZE*2, 0+BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': CLUSTERS['horizon_iss'], 'color': 'x', 'position': '2'}, 
        {'origin': [1080*3+BORDER_SIZE*3-15, 0+BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20170726-004001', '20170723-231002'], 'color': 'x', 'position': '3'},
        {'origin': [1080*2+BORDER_SIZE*2, 720-15+2*BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20190101-220001', '20180625-140001', '20180617-112001'], 'color': 'x', 'position': '4'}, 
        {'origin': [1080*3+BORDER_SIZE*3-15, 720-15+2*BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': CLUSTERS['horizon_only'], 'color': 'x', 'position': '5'},

        {'origin': [0+BORDER_SIZE, 720*2+BORDER_SIZE*2], 'size': [1080-15, 720-15], 'cluster': CLUSTERS['vertical'], 'color': 'x', 'position': '6'}, 
        {'origin': [1080+BORDER_SIZE*2-15, 720*2+BORDER_SIZE*2], 'size': [1080-15, 720-15], 'cluster': CLUSTERS['vertical'], 'color': 'x', 'position': '7'}, 
        {'origin': [0+BORDER_SIZE, 720*3-15+BORDER_SIZE*3], 'size': [1080-15, 720-15], 'cluster': CLUSTERS['vertical'], 'color': 'x', 'position': '8'},
        {'origin': [1080+BORDER_SIZE*2-15, 720*3-15+BORDER_SIZE*3], 'size': [1080-15, 720-15], 'cluster': CLUSTERS['vertical'], 'color': 'x', 'position': '9'},
        {'origin': [1080*2+BORDER_SIZE*3-15*2, 720*2-15*2+BORDER_SIZE*3], 'size': [1080*2, 720*2], 'cluster': CLUSTERS['horizon_only'][4:], 'color': '2x', 'position': '10'}
    ]
},
{
    "version": "3",
    "size":  {
        'width': O_WIDTH*4 + BORDER_SIZE*3, 
        'height': O_HEIGHT*4 + BORDER_SIZE*3,
        'channel': 3
    },
    "ARGS_SQL": (1, 'simple_conv', 'model_colab'),
    "DECOUPAGE": [
        {'origin': [0+BORDER_SIZE, 0+BORDER_SIZE], 'size': [1080*2, 720*2], 'cluster': None, 'picture_ids' : ['20170815-130001'], 'directory': 'data/isr/output/version_3/rrdn-C4-D3-G32-G032-T10-x4/2019-12-16_2245', 'color': '2x', 'position': '1'},
        {'origin': [1080*2+BORDER_SIZE*2, 0+BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20170424-133001'], 'color': 'x', 'position': '2'}, 
        {'origin': [1080*3+BORDER_SIZE*3-15, 0+BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20170726-004001'], 'color': 'x', 'position': '3'},
        {'origin': [1080*2+BORDER_SIZE*2, 720-15+2*BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20180625-140001', '20180617-112001'], 'color': 'x', 'position': '4'}, 
        {'origin': [1080*3+BORDER_SIZE*3-15, 720-15+2*BORDER_SIZE], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20180731-000001'], 'color': 'x', 'position': '5'},

        {'origin': [0+BORDER_SIZE, 720*2+BORDER_SIZE*2], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20180902-100001'], 'directory': 'data/isr/output/version_3/rrdn-C4-D3-G32-G032-T10-x4/2019-12-16_2245', 'color': 'x', 'position': '6'}, 
        {'origin': [1080+BORDER_SIZE*2-15, 720*2+BORDER_SIZE*2], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20181129-080002'], 'color': 'x', 'position': '7'}, 
        {'origin': [0+BORDER_SIZE, 720*3-15+BORDER_SIZE*3], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20170420-195001'], 'color': 'x', 'position': '8'},
        {'origin': [1080+BORDER_SIZE*2-15, 720*3-15+BORDER_SIZE*3], 'size': [1080-15, 720-15], 'cluster': None, 'picture_ids': ['20180118-032001'], 'color': 'x', 'position': '9'},
        {'origin': [1080*2+BORDER_SIZE*3-15*2, 720*2-15*2+BORDER_SIZE*3], 'size': [1080*2, 720*2], 'cluster': None, 'picture_ids': ['20180613-102001'], 'color': '2x', 'position': '10'}
    ]
}]


##
## Fonctions
##

def get_pictures_df(config, db_manager, poster_config):

    
    req_sql = "SELECT * FROM iss.pictures_embedding WHERE clustering_version = %s AND clustering_model_type = %s AND clustering_model_name = %s"

    pictures_df = db_manager.select_df(req_sql, poster_config['ARGS_SQL'])

    return pictures_df


def create_empty_poster(config, poster_config):
    poster = np.zeros((
        poster_config['size']['height'],
        poster_config['size']['width'],
        poster_config['size']['channel']
    ))

    return poster


def create_poster_template_picture(config, poster_config, poster):
    
    font = ImageFont.truetype("LiberationMono-Regular.ttf", 70)

    for part in poster_config['DECOUPAGE']:
        rgb_values = COLORS[part['color']]['rgb']
        for c, v in enumerate(rgb_values):
            poster[part['origin'][1]:(part['origin'][1]+part['size'][1]), part['origin'][0]:(part['origin'][0]+part['size'][0]), c] = v
    poster = poster.astype('uint8')
    img = PIL.Image.fromarray(poster, 'RGB')
    d = ImageDraw.Draw(img)

    for part in poster_config['DECOUPAGE']:
        d.text((part['origin'][0],part['origin'][1]), "position %s" % (part['position']), fill=(0,0,0), font=font)

    return np.array(img)


def create_poster_picture(config, poster_config, poster, pictures_df):

    pictures_id_positions = []
    for part in poster_config['DECOUPAGE']:

        picture_id = None
        directory = config.get('directory')['collections']
        if 'directory' in part.keys() and part['directory']:
            directory = os.path.join(config.get('directory')['project_dir'], part['directory'])

        if part['cluster']:
            selection = pictures_df.loc[pictures_df.label.isin(part['cluster'])].sample(1)
            pictures_df = pictures_df.drop(selection.index)

            picture_id = selection.pictures_id.values[0]
        else:
            picture_id = random.choice(part['picture_ids'])

        pictures_id_positions.append({'pictures_id': picture_id, 'position': part['position']})
        
        if part['size'][0] < O_WIDTH or part['size'][1] < O_HEIGHT:
            part_array = Tools.read_np_picture(os.path.join(directory, "%s.jpg" % picture_id ), target_size = (O_HEIGHT, O_WIDTH))
            index_col = np.floor((O_WIDTH - part['size'][0])/2).astype(int)
            index_row = np.floor((O_HEIGHT - part['size'][1])/2).astype(int)
            part_array = part_array[index_col:(index_col+part['size'][1]), index_row:(index_row+part['size'][0]), :].copy()
        else:
            part_array = Tools.read_np_picture(os.path.join(directory, "%s.jpg" % picture_id ), target_size = (part['size'][1], part['size'][0]))

        poster[part['origin'][1]:(part['origin'][1]+part['size'][1]), part['origin'][0]:(part['origin'][0]+part['size'][0]), :] = part_array

    return poster, pictures_id_positions


def save_poster_picture(config, poster_config, pictures_id_positions, db_manager, poster_id = None):
    poster_id = poster_id if poster_id else uuid.uuid1().hex
    array = []
    for p_pos in pictures_id_positions:
        array.append((
            poster_id,
            poster_config['version'],
            p_pos['position'],
            p_pos['pictures_id']
        ))

    db_manager.insert_row_poster(array)

    return poster_id


def write_poster_picture(config, poster_config, poster, poster_id = None):

    poster_id = poster_id if poster_id else uuid.uuid1()
    img = Tools.display_one_picture(poster)
    img.save(os.path.join(CONFIG.get('directory')['data_dir'], 'posters', "version_%s_%s.jpg" % (poster_config['version'], poster_id)), "JPEG")
    
    return poster_id


##
## Main
##

@click.command()
@click.option('--config-id', default=1, show_default=True, type=int)
@click.option('--generate', default=1, show_default=True, type=int)
@click.option('--poster-id', default=None, show_default=True, type=str)
def main(config_id, generate, poster_id):
    
    N_GENERATE = generate
    POSTER_ID = poster_id
    
    db_manager = Tools.create_db_manager(CONFIG)

    ## creation de la base
    db_manager.create_posters_table()

    ##
    poster_config = POSTERS_CONFIG[config_id]
    
    ##
    pictures_df = get_pictures_df(CONFIG, db_manager, poster_config)

    ##
    poster = create_empty_poster(CONFIG, poster_config)

    ##
    poster = create_poster_template_picture(CONFIG, poster_config, poster)

    ##
    write_poster_picture(CONFIG, poster_config, poster, poster_id = "template")

    for i in range(N_GENERATE):
        print("%s / %s" % (i+1, N_GENERATE))
        ##
        poster, p_pos = create_poster_picture(CONFIG, poster_config, poster, pictures_df)

        ##
        poster_id = save_poster_picture(CONFIG, poster_config, p_pos, db_manager, POSTER_ID)

        ##
        write_poster_picture(CONFIG, poster_config, poster, poster_id)


if __name__  == '__main__':
    main()
