import os
import base64
import pandas as pd
import numpy as np
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

from iss.init_config import CONFIG
from iss.tools import Tools


SPRITE_NB_LIGNE = 145
SPRITE_NB_COLONNE = 100
TARGET_SIZE_WIDTH = 48*2
TARGET_SIZE_HEIGHT = 27*2
LIMIT = 14499

def request_data(config, db_manager):

    sql = """
    SELECT 
        v1.pictures_id,

        v1.pictures_x as v1_x,
        v1.pictures_y as v1_y,
        CAST(v1.label AS CHAR) as v1_label,

        v2.pictures_x as v2_x,
        v2.pictures_y as v2_y,
        CAST(v2.label AS CHAR) as v2_label,

        v3.pictures_x as v3_x,
        v3.pictures_y as v3_y,
        CAST(v3.label AS CHAR) as v3_label,

        loc.pictures_timestamp,
        loc.pictures_location_text,
        loc.pictures_latitude,
        loc.pictures_longitude

    FROM iss.pictures_embedding AS v1

    INNER JOIN iss.pictures_embedding v2
    ON v1.pictures_id = v2.pictures_id
    AND v2.clustering_type = v1.clustering_type
    AND v2.clustering_model_type = v1.clustering_model_type
    AND v2.clustering_model_name = v2.clustering_model_name
    AND v2.clustering_version = 2

    INNER JOIN iss.pictures_embedding v3
    ON v1.pictures_id = v3.pictures_id
    AND v3.clustering_type = v1.clustering_type
    AND v3.clustering_model_type = v1.clustering_model_type
    AND v3.clustering_model_name = v1.clustering_model_name
    AND v3.clustering_version = 3

    LEFT JOIN iss.pictures_location loc
    ON loc.pictures_id = v1.pictures_id

    WHERE v1.clustering_version = %s
    ORDER BY pictures_id ASC LIMIT %s"""

    db_manager.cursor.execute(sql, (1, LIMIT))
    results = db_manager.cursor.fetchall()
     
    return pd.DataFrame(results, columns=db_manager.cursor.column_names)


def create_sprite(config, df):

    images_array = [Tools.read_np_picture(os.path.join(config.get('directory')['collections'], "%s.jpg" % picture_id), target_size = (TARGET_SIZE_HEIGHT, TARGET_SIZE_WIDTH)) for picture_id in df['pictures_id']]
    sprite = np.zeros((TARGET_SIZE_HEIGHT*SPRITE_NB_LIGNE, TARGET_SIZE_WIDTH*SPRITE_NB_COLONNE, 3))
    index = 0
    for i in range(SPRITE_NB_LIGNE):
        for j in range(SPRITE_NB_COLONNE):
            sprite[(i*TARGET_SIZE_HEIGHT):(i+1)*TARGET_SIZE_HEIGHT, (j*TARGET_SIZE_WIDTH):(j+1)*TARGET_SIZE_WIDTH, :] = images_array[index]
            index += 1
            if index >= len(images_array):
                break
        if index >= len(images_array):
            break

    img = Tools.display_one_picture(sprite)
    return img


def generate_facets(config, df):

    proto = GenericFeatureStatisticsGenerator().ProtoFromDataFrames([{'name': 'facets-iss', 'table': df}])
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")

    HTML_TEMPLATE = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
            <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
            <facets-overview id="elem"></facets-overview>
            <script>
            document.querySelector("#elem").protoInput = "{protostr}";
            </script>"""
    html = HTML_TEMPLATE.format(protostr=protostr)

    return html

def generate_facets_dive(config, df, relative_sprite_path):

    jsonstr = df.to_json(orient = 'records')
    HTML_TEMPLATE = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
            <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
            <facets-dive id="elem" height="600" cross-origin="anonymous" sprite-image-width="{sprite_width}" sprite-image-height="{sprite_height}">
            </facets-dive>
            <script>
            var data = {jsonstr};
            var atlas_url = "{atlas_url}";
            document.querySelector("#elem").data = data;
            document.querySelector("#elem").atlasUrl = atlas_url;
            </script>"""
    html = HTML_TEMPLATE.format(jsonstr=jsonstr, atlas_url = relative_sprite_path, sprite_width=TARGET_SIZE_WIDTH, sprite_height=TARGET_SIZE_HEIGHT)
    
    return html


def main():

    ## db manager
    db_manager = Tools.create_db_manager(CONFIG)

    ## request data
    df = request_data(CONFIG, db_manager)

    ## create sprite
    sprite = create_sprite(CONFIG, df)

    ## save sprite
    sprite.save(os.path.join(CONFIG.get('directory')['reports'], 'figures', 'sprite_altas.png'), "PNG")

    ## generate facets
    html_facets = generate_facets(CONFIG, df)
    with open(os.path.join(CONFIG.get('directory')['reports'], 'facets.html'),'w') as f:
        f.write(html_facets)

    ## generate facets-dive
    html_facets_dive = generate_facets_dive(CONFIG, df, './figures/sprite_altas.png')
    with open(os.path.join(CONFIG.get('directory')['reports'], 'facets-dive.html'), 'w') as f:
            f.write(html_facets_dive)


if __name__ == '__main__':
    main()