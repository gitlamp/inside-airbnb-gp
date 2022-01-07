import psycopg2
import os
import pandas as pd
import pandas.io.sql as sqlio
from configparser import ConfigParser
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
from sklearn.cluster import KMeans
from darksky import forecast

dir_path = os.path.dirname(__file__)


def read_config(filename=f'{dir_path}/config.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    conf = {}
    if parser.has_section(section) and section == 'postgresql':
        params = parser.items(section)
        for param in params:
            conf[param[0]] = param[1]
    elif parser.has_section(section) and section == 'openweathermap':
        key = parser.get('openweathermap', 'key').split(',')
        conf['owm_key'] = key
    elif parser.has_section(section) and section == 'darksky':
        key = parser.get('darksky', 'key').split(',')
        conf['dksky_key'] = key
    elif parser.has_section(section) and section == 'airbnb':
        params = parser.items(section)
        for param in params:
            links = param[1].split(',')
            conf[param[0]] = list(filter(None, links))
    else:
        raise Exception(f'Section {section} not found in {filename}')

    return conf


def gen_cluster(df, n_clusters=1000):
    try:
        print('Calculating clusters ...')
        model = KMeans(n_clusters=n_clusters)
        X = df[['latitude', 'longitude']]
        kmeans = model.fit(X)
        df['cluster'] = kmeans.labels_
        pred = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
        res = pd.merge(df, pred, left_on='cluster', right_on=pred.index, suffixes=('', '_cent'))
        res.drop(['latitude', 'longitude'], axis=1, inplace=True)
        print('Clusters are calculated!')
        return res

    except Exception as err:
        print(err)


@sleep_and_retry
@limits(calls=500, period=24*60*60)
def call_dksky_api(key, lat, long, time):

    try:
        res = {}
        r = forecast(key, lat, long, time, units='si')
        daily = r.daily[0]
        res['temperatureAvg'] = (daily.temperatureMax+daily.temperatureMin)/2
        res['icon'] = daily.icon if hasattr(daily, 'icon') else None
        res['cloudCover'] = daily.cloudCover if hasattr(daily, 'cloudCover') else None
        res['ozone'] = daily.ozone if hasattr(daily, 'ozone') else None
        res['humidity'] = daily.humidity if hasattr(daily, 'humidity') else None
        res['pressure'] = daily.pressure if hasattr(daily, 'pressure') else None
        res['summary'] = daily.summary if hasattr(daily, 'summary') else None
        res['time'] = daily.time if hasattr(daily, 'time') else None
        res['windSpeed'] = daily.windSpeed if hasattr(daily, 'windSpeed') else None

        return res
    except Exception as err:
        print(err)


def connect():
    conn = None
    try:
        params = read_config()
        print('Connecting to database ...')
        conn = psycopg2.connect(**params)

        cur = conn.cursor()
        query = """
        CREATE TABLE IF NOT EXISTS owm
        (
            id         serial
                CONSTRAINT owm_pk
                    PRIMARY KEY,
            listing_id integer NOT NULL,
            clouds     integer,
            status     text,
            temp       integer,
            humidity   integer,
            co         integer,
            o3         integer,
            no2        integer,
            so2        integer
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS owm_id_uindex
            ON owm (id);
        
        CREATE TABLE IF NOT EXISTS cluster
        (
            id         serial
                CONSTRAINT cluster_sk 
                    PRIMARY KEY,
            airbnb_id integer NOT NULL,
            cluster    integer,
            latitude  numeric,
            longitude  numeric
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS cluster_id_uindex
            ON cluster (id);
            
        CREATE TABLE IF NOT EXISTS dksky
        (
            id             serial
                CONSTRAINT dksky_sk
                    PRIMARY KEY,
            airbnb_id      integer NOT NULL,
            last_scraped   date,
            icon           text,
            cloudCover     float4,
            ozone          float4,
            humidity       float4,
            temperatureAvg float4,
            pressure       float4,
            summary        text,
            time           bigint,
            windSpeed      float4
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS dksky_id_uindex
            ON dksky (id);
        """
        cur.execute(query)
        conn.commit()

        # generate clusters
        query = """
                SELECT DISTINCT airbnb_id,
                                latitude,
                                longitude
                FROM amsterdam
                UNION
                SELECT DISTINCT airbnb_id,
                                latitude,
                                longitude
                FROM athens
                UNION
                SELECT DISTINCT airbnb_id,
                                latitude,
                                longitude
                FROM berlin
                """
        location = sqlio.read_sql(query, conn)
        query = """
                SELECT COUNT(*)
                FROM (SELECT DISTINCT calendar_last_scraped
                      FROM amsterdam
                      UNION
                      SELECT DISTINCT calendar_last_scraped
                      FROM athens
                      UNION
                      SELECT DISTINCT calendar_last_scraped
                      FROM berlin) all_cities
                """
        cur.execute(query)
        n_clusters = 200
        clusters = gen_cluster(location, n_clusters)
        tuples = list(clusters.itertuples(False, None))

        query = """
                INSERT INTO cluster(airbnb_id, cluster, latitude, longitude)
                VALUES(%s,%s,%s,%s)
                """
        cur.executemany(query, tuples)
        conn.commit()

        # Darksky
        query = """
                SELECT ARRAY_AGG(c.airbnb_id), c.latitude, c.longitude, l.last_scraped::timestamp
                FROM cluster c
                         LEFT JOIN (
                    SELECT airbnb_id, last_scraped
                    FROM amsterdam
                    UNION
                    SELECT airbnb_id, last_scraped
                    FROM athens
                    UNION
                    SELECT airbnb_id, last_scraped
                    FROM berlin) l ON l.airbnb_id = c.airbnb_id
                    WHERE c.airbnb_id NOT IN (
                    SELECT airbnb_id FROM dksky
                    )
                GROUP BY c.latitude, c.longitude, l.last_scraped;
                """
        cur.execute(query)
        row = cur.fetchall()
        key = read_config(section='darksky')['dksky_key'][0]

        with tqdm(total=len(row), desc='Calling Darksky api', unit='req') as dbar:
            for list_id, lat, long, last_scraped in row:
                time = last_scraped.isoformat()
                res = call_dksky_api(key, lat, long, time)
                for id in list_id:
                    query = """
                            INSERT INTO dksky(airbnb_id,
                                            last_scraped,
                                            icon,
                                            cloudCover,
                                            ozone,
                                            humidity,
                                            temperatureAvg,
                                            pressure,
                                            summary,
                                            time,
                                            windSpeed)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """
                    values = (id,
                              last_scraped,
                              res['icon'],
                              res['cloudCover'],
                              res['ozone'],
                              res['humidity'],
                              res['temperatureAvg'],
                              res['pressure'],
                              res['summary'],
                              res['time'],
                              res['windSpeed'])
                    cur.execute(query, values)
                conn.commit()
                dbar.update(1)

        dbar.close()
        cur.close()

    except(Exception, psycopg2.DatabaseError) as err:
        print(err)
    finally:
        if conn is not None:
            conn.close()
            print('Connection closed.')


if __name__ == '__main__':
    connect()
