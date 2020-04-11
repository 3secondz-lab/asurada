import numpy as np
from scipy import signal
import pandas as pd

class DataHelper(object):
    def __init__(self, dataframe=None, timestamp=None, lat=None, lon=None, localX=None, localY=None, speed=None, heading=None):
        self.origin = np.array([None,None])

        if not dataframe is None:
            self.df = dataframe
            timestamp = dataframe['TimeStamp'].values
            lat = dataframe['PosLat'].values
            lon = dataframe['PosLon'].values
            localX = dataframe['PosLocalX'].values
            localY = dataframe['PosLocalY'].values
            speed = dataframe['GPS_Speed'].values
            heading = dataframe['AngleTrack'].values

            self.set_position(localX, localY)

        elif not hasattr(self, 'localX') or not hasattr(self, 'localY'):
            if not (localX is None or localY is None):
                self.set_position(localX, localY)

            elif not (lat is None or lon is None):
                self.set_lat(lat)
                self.set_lon(lon)
                self.set_position()

        if not speed is None:
            self.set_speed(speed)

        if not heading is None:
            self.set_heading(heading)

        if not timestamp is None:
            self.set_timestamp(timestamp)

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
        self.distance, self.curvature = self.station_curvature(self.localX, self.localY)
        
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
    def station_curvature(localX, localY):
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
        k = signal.medfilt(k, 9)

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