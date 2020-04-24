import numpy as np
from scipy import signal
import pandas as pd

class DataHelper(object):
    def __init__(self, dataframe, timestamp=None, speed=None,
                 localX=None, localY=None, lat=None, lon=None,
                 heading=None, medfilt=15):

        self.origin = np.array([None,None])
        self.medfilt = medfilt

        self.df = dataframe  # necessary to make an output dictionary of 'get_preview'

        # If there is no input for each variable required for DataHelper operation,
        # it is taken from the dataframe and used.

        # timestamp
        if timestamp is None:  # if a variable 'timestamp' is not None, we can execute the function 'set_timestamp()' immediately'
            # if not, we have to find the values from df.
            assert 'TimeStamp' in self.df.columns, 'Necessary Field. If your dataframe doesn\'t have TimeStamp, you can input it by timestamp= '
            timestamp = self.df['TimeStamp'].values
        self.set_timestamp(timestamp)

        # speed
        if speed is None:
            assert 'GPS_Speed' in dataframe.columns, 'Necessary Field. If your dataframe doesn\'t have GPS_Speed, you can input it by speed= '
            speed = self.df['GPS_Speed'].values
        self.set_speed(speed)

        # localX/ localY
        if localX is None or localY is None:
            if lat is None or lon is None:
                if 'PosLocalX' in self.df.columns and 'PosLocalY' in self.df.columns:  # when df has localX/Y
                    localX = self.df['PosLocalX'].values
                    localY = self.df['PosLocalY'].values
                    self.set_position(localX, localY)
                elif 'PosLat' in self.df.columns and 'PosLon' in self.df.columns:  # when df has lat/lon instead of localX/Y
                    lat = self.df['PosLat'].values
                    lon = self.df['PosLon'].values
                    self.set_lat(lat)
                    self.set_lon(lon)
                    self.set_position()
                else:
                    raise NameError('Either (PosLocalX, PosLocalY) or (PosLat, PosLon) must be essential. \
                                     You can input these by (localX=, localY=) or (lat=, lon=)')
            else:
                self.set_lat(lat)
                self.set_lon(lon)
                self.set_position()
        if not hasattr(self, 'localX') or not hasattr(self, 'localY'):
            self.set_position(localX, localY)

        # heading
        if heading is None:
            if 'AngleTrack' in self.df:
                heading = self.df['AngleTrack'].values
            else:
                heading = self.get_heading()
        self.set_heading(heading)

    def set_lat(self, lat):
        self.lat = self.assign_checker(lat)
        self.origin[0] = lat[0]

    def set_lon(self, lon):
        self.lon = self.assign_checker(lon)
        self.origin[1] = lon[0]

    def set_position(self, localX=None, localY=None):
        if not localX is None and not localY is None:
            self.localX = self.assign_checker(localX)
            self.localY = self.assign_checker(localY)
        else:
            self.localX, self.localY = self.geo_to_lin(self.lat, self.lon, self.origin)
        self.distance, self.curvature = self.station_curvature(self.localX, self.localY, self.medfilt)

    def set_speed(self, speed):
        self.speed = self.assign_checker(speed)

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        self.interval = np.diff(timestamp)

    def set_heading(self, heading):
        self.heading = self.assign_checker(heading)
        if np.max(heading) > 2*np.pi:
            self.heading = np.deg2rad(self.heading)

    def set_preview_distance(self, preview_distance):
        self.preview_distance = preview_distance

    def set_preview_time(self, preview_time):
        self.preview_time = preview_time

    def get_index(self, localX, localY):
        if hasattr(localX, '__len__') or hasattr(localY, '__len__'):
            #TODO
            pass
        else:
            #TODO
            pass

    def get_position(self, ind=None):
        if ind:
            return self.localX[ind], self.localY[ind]
        else:
            return self.localX, self.localY

    def get_preview(self, ind, method = 'DISTANCE'):
        if method is 'DISTANCE':
            if not hasattr(self, 'preview_distance'):
                print('class DataHelper -> set preview distance first : DataHelper.set_preview_distance(distance)')
                raise ValueError
            window = list(range(ind, self.nearest(self.distance[ind]+self.preview_distance, self.distance)))
            if len(window) < 2:
                window = list(range(ind,ind+1))

        elif method is 'TIME':
            if not hasattr(self, 'preview_time'):
                print('class DataHelper -> set preview time first : DataHelper.set_preview_time(time)')
                raise ValueError
            window = list(range(ind, self.nearest(self.timestamp[ind]+self.preview_time, self.timestamp)))
            if len(window) < 2:
                window = list(range(ind, ind+1))

        else:
            print('class DataHelper -> Supported preview methods : {DISTANCE, TIME}')
            raise ValueError

        res = {}
        if hasattr(self, 'df'):
            for item in self.df.columns:
                res[item] = self.df[item][window].to_numpy()
            res['PreviewX'], res['PreviewY'] = self.get_preview_plane(window)
            res['Curvature'] = self.curvature[window]
            res['Distance'] = self.distance[window] - self.distance[ind]
            res['AngleTrack'] = self.heading[window] - self.heading[ind]

            #### TODO ####
            '''
            define output when data input is not in the form of pandas.dataframe
            '''
        # for item in res.keys():
        #     res[item] = np.interp(rwindow, window, res[item], res[item][0], res[item][-1])
        return res

    def get_preview_plane(self, window):
        #TODO : Condition Check
        ind = window[0]
        return self.transform_plane(self.localX[window]-self.localX[ind], self.localY[window]-self.localY[ind], self.heading[ind])

    def get_heading(self):
        return np.mod(np.pi/2 - np.arctan2(np.gradient(self.localY), np.gradient(self.localX)), 2*np.pi)

    @staticmethod
    def assign_checker(arg):
        if isinstance(arg, np.ndarray):
            res = arg
        elif isinstance(arg, pd.dataframe):
            res = arg.to_numpy()
        else:
            res = np.ndarray(arg)

        return res


    @staticmethod
    def transform_plane(localX, localY, heading):

        m = np.array([[np.sin(heading), np.cos(heading)], [np.cos(heading), -np.sin(heading)]])

        preview = m.dot(np.vstack(([localX],[localY])))

        return preview[0], preview[1]


    @staticmethod
    def nearest(origin, dist):
        idx = np.abs(origin - dist).argmin()
        return idx

    @staticmethod
    def geo_to_lin(lat, lon, origin):
        _R0 = 6378137.0
        _E=1/298.257223563
        _RAD_DEGREE = np.pi/180.0
        delta_lon = lon - origin[1]
        delta_lat = lat - origin[0]
        _Rn = _R0 *(1-_E**2)/((1-(_E**2)*(np.sin(origin[0]*_RAD_DEGREE)**2))**(1.5))
        _Re = _R0/((1-(_E**2)*(np.sin(origin[0]*_RAD_DEGREE)**2))**(0.5))
        _Ra = _Re*_Rn / np.sqrt( (_Re**2)*(np.sin(origin[0]*_RAD_DEGREE)**2) + (_Rn**2)*(np.cos(origin[0]*_RAD_DEGREE)**2) )
        localX = _Ra * np.cos(origin[0]*_RAD_DEGREE) * delta_lon * np.pi/180.0
        localY = _Rn * delta_lat * np.pi/180.0

        return localX , localY

    @staticmethod
    def lin_to_geo(localX, localY, origin):
        _R0 = 6378137.0
        _E=1/298.257223563
        _RAD_DEGREE = np.pi/180.0
        _Rn = _R0*(1-_E**2)/((1-_E**2 * (np.sin(origin[0]*_RAD_DEGREE)**2))**(1.5))
        _Re = _R0/((1-_E**2 *(np.sin(origin[0])**2))**(0.5))
        _Ra = _Re*_Rn / np.sqrt( _Re**2*np.sin(origin[0]*_RAD_DEGREE)**2 + _Rn**2*np.cos(origin[0]*_RAD_DEGREE)**2 )

        delta_lon = localX / ( _Ra * np.cos(origin[0]*_RAD_DEGREE) * np.pi /180 )
        delta_lat = localY / ( _Rn * np.pi / 180 )

        lon =  delta_lon + origin[1]
        lat =  delta_lat + origin[0]

        return lon , lat

    @staticmethod
    def station_curvature(localX, localY, medfilt):
        x = localX
        y = localY
        dx = np.append([0], np.diff(x))
        dy = np.append([0], np.diff(y))
        s = np.zeros(len(dx))
        k = np.zeros(len(dx))
        for i in range(len(dx)):
            s[i] = np.sqrt(dx[i]**2 + dy[i]**2)

        for i in range(1,len(dx)-2):
            if dx[i+2] == 0 or dy[i+2] == 0 or dx[i+1] == 0 or dy[i+1] == 0:
                k[i+1] = k[i]
                continue
            k_1 = ((x[i+1]-x[i])/np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2) + (x[i+2]-x[i+1])/np.sqrt((x[i+2]-x[i+1])**2+(y[i+2]-y[i+1])**2))
            k_2 = ((y[i+2]-y[i+1])/np.sqrt((x[i+2]-x[i+1])**2+(y[i+2]-y[i+1])**2) - (y[i+1]-y[i])/np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
            k_3 = (np.sqrt((x[i+2]-x[i+1])**2+(y[i+2]-y[i+1])**2)+np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
            k_4 = ((y[i+1]-y[i])/np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2) + (y[i+2]-y[i+1])/np.sqrt((x[i+2]-x[i+1])**2+(y[i+2]-y[i+1])**2))
            k_5 = ((x[i+2]-x[i+1])/np.sqrt((x[i+2]-x[i+1])**2+(y[i+2]-y[i+1])**2) - (x[i+1]-x[i])/np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
            k_6 = (np.sqrt((x[i+2]-x[i+1])**2+(y[i+2]-y[i+1])**2)+ np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
            k[i+1] =  k_1 * k_2/ k_3 - k_4 * k_5 / k_6


        k[0] = k[1]
        k[len(k)-1] = k[len(k) - 2]
        k = np.array(k)
        k[k>0.5] = 0.5
        k[k<-0.5] = -0.5
        # k = signal.medfilt(k, 9)
        k = signal.medfilt(k, medfilt)

        return np.cumsum(s), k


    @staticmethod
    def unwrap(lst):
        lst1 = np.rad2deg(np.unwrap(np.deg2rad(lst)))

        return lst1

    @staticmethod
    def movmean(a, n):
        if n%2 == 1 :
            front = n//2
            back = n//2
        else :
            front = n//2
            back = n//2 -1
        ret = np.zeros(len(a))
        len_a = len(a)
        for i in range(len(a)):
            if i < front :
                val = np.mean(a[:i+back+1])
            elif i>len_a -back -1:
                val = np.mean(a[i-front:])
            else :
                val = np.mean(a[i-front:i+back+1])
            ret[i] = val
        return ret

if __name__ == "__main__":
    ''' DataHelper Example Code (datafile: std_xxx.csv) '''

    import pdb
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    datafile = 'std_001.csv'
    df = pd.read_csv(datafile)

    dh = DataHelper(df)
    print('List of local variables of DataHelper:\n', sorted([*vars(dh).keys()]))
    print(' ')

    dh.set_preview_time(1)
    ind = 1
    res = dh.get_preview(ind, method='TIME')  # df fields + [PreviewX, PreviewY, Curvature, Distance]
    print('Preview information field list:\n', sorted([*res.keys()]))

    ''' Time based preview '''
    dh.set_preview_time(10.0)
    ind = 1000  # Index at which preview function is executed
    res = dh.get_preview(ind, method='TIME')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    ax1.plot(res['PosLocalX'], res['PosLocalY'])
    ax2.plot(res['TimeStamp'],res['Curvature'])
    ax3.plot(res['TimeStamp'],res['GPS_Speed'])

    ax1.set_xlabel('PosLocalX')
    ax1.set_ylabel('PosLocalY')
    ax2.set_xlabel('TimeStamp')
    ax2.set_ylabel('curvature')
    ax3.set_xlabel('TimeStamp')
    ax3.set_ylabel('GPS_Speed')

    fig.suptitle('Time based preview [{}s]'.format(dh.preview_time))


    ''' Distance based preview '''
    dh.set_preview_distance(250.0) # 250m preview
    ind = 1000 # Index at which preview function is executed
    res = dh.get_preview(ind, method='DISTANCE')

    fig2, (ax21, ax22, ax23) = plt.subplots(1, 3, figsize=(20, 6))
    ax21.plot(res['PosLocalX'], res['PosLocalY'])
    ax22.plot(res['TimeStamp'],res['Curvature'])
    ax23.plot(res['TimeStamp'],res['GPS_Speed'])

    ax21.set_xlabel('PosLocalX')
    ax21.set_ylabel('PosLocalY')
    ax22.set_xlabel('TimeStamp')
    ax22.set_ylabel('curvature')
    ax23.set_xlabel('TimeStamp')
    ax23.set_ylabel('GPS_Speed')

    fig2.suptitle('Distance based preview [{}m]'.format(dh.preview_distance))

    plt.show()
