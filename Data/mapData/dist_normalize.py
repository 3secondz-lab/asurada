import pdb
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# distance, inner_xy, outer_xy, 

def getNextPosition(ref_x, ref_y, ref_ind, curx, cury):
	nrx, nry =  ref_x[ref_ind], ref_y[ref_ind]  # next ref point

	theta = np.arctan(abs((nry-cury)/(nrx-curx)))
	delx = np.cos(theta)
	dely = np.sin(theta)

	if nrx <= curx:
		nnx = curx - delx
	else:
		nnx = curx + delx 

	if nry <= cury:
		nny = cury - dely
	else:
		nny = cury + dely
	return nnx, nny

def getDistance(ax, ay, bx, by):
	return np.sqrt((ax-bx)**2 + (ay-by)**2)

parser = argparse.ArgumentParser()
parser.add_argument('--circuit', '-c')

args = parser.parse_args()
mapName = args.circuit

df_map = pd.read_csv('{}.csv'.format(mapName))

inner_x = df_map['inner_x'].values
inner_y = df_map['inner_y'].values
outer_x = df_map['outer_x'].values
outer_y = df_map['outer_y'].values
center_x = (inner_x + outer_x)/2 
center_y = (inner_y + outer_y)/2

# 원래 쓰던 전처리 코드에서 inner_x랑 outer_x를 이용해서 center_xy를 계산하는데, 
# 1m로 normalize를 하려다보니, inner_x 선에서의 1m와 outer_x에서의 1m가 또 다를거 같아서, 
# 차라리, map data의 inner_x랑 outer_x를 이용해서 center_xy를 계산하고, 
# 이 값을 전처리에서 inner_, outer_, 없이 center_xy를 바로 가져다 쓰는게 제일 나을거 같음... 

#####
# 그럼 center_xy랑 dist만 저장하고, 이걸 가져다 쓰는 방법으로 전처리 코드를 바꿔야겠다.
# 그리고 출발점, 도착점 연결하는거 잘 하고. (중간에 걍 1m로 채워넣기!, 아니면 걍 연결하기(0.01 뭐 이런 경우에))
#####

dist = df_map['distance'].values
# dist_diff = np.diff(dist)

# inner_x_norm = [inner_x[0]]
# inner_y_norm = [inner_y[0]]
# outer_x_norm = [outer_x[0]]
# outer_y_norm = [outer_y[0]]
nx, ny = center_x[0], center_y[0]
center_x_norm = [nx]
center_y_norm = [ny]
dist_norm = [0]

ri = 0  # reference index
while(True):
	if ri == len(dist)-1:  # 한바퀴를 다 돌았음. 그렇다면, 출발점이랑 거리가 있는 경우, 연결.
		print(nx, ny)
		temp_dist = getDistance(nx, ny, center_x[0], center_y[0])
		print('Distance betw. [0], [-1]: {}'.format(temp_dist))
		# pdb.set_trace()
		if temp_dist > 1.0:
			nnx, nny = getNextPosition(center_x, center_y, 0, nx, ny)
			center_x_norm.append(nnx)
			center_y_norm.append(nny)
			dist_nnn = getDistance(nx, ny, nnx, nny)
			dist_norm.append(dist_norm[-1] + dist_nnn)
			nx = nnx
			ny = nny
			continue

		else:
			break 
			# map에 따라서, center_x/y_norm[0]과 center_x/y_norm[-1]의 거리가
			# 1보다 작은 경우가 있는데, 
			# imola			0.57
			# mugello		0.016
			# magione		0.98
			# spa			0.039
			# monza			0.83	
			# nordschleife 	0.66
			# 그냥 이 사이 거리도 1m로 해버려도, road shape에 끼치는 영향은 작을거 같다.

	for j in range(ri, len(dist)):
		# pdb.set_trace()
		# dist_nr = np.sqrt((nx-center_x[j])**2 + (ny-center_y[j])**2)  # new to ref
		dist_nr = getDistance(nx, ny, center_x[j], center_y[j])
		# print(j, dist_nr)

		if dist_nr > 1.0:
			# print(i, j, dist_nr)
			ri = j  # j가 reference로 계산이 되고나서, ri값이 update됨.
			break

	nnx, nny = getNextPosition(center_x, center_y, ri, nx, ny)

	center_x_norm.append(nnx)
	center_y_norm.append(nny)

	# dist_nnn = np.sqrt((nnx-nx)**2+(nny-ny)**2)
	dist_nnn = getDistance(nx, ny, nnx, nny)
	dist_norm.append(dist_norm[-1] + dist_nnn)
	
	nx = nnx 
	ny = nny
	# print(dist_nnn, nx, ny)

	# plt.figure()
	# plt.plot(center_x[0], center_y[0], 'rx')
	# plt.plot(center_x_norm, center_y_norm, 'r.-')
	# plt.plot(center_x[:ri+3], center_y[:ri+3], 'b.-')
	# plt.show()
	# pdb.set_trace()


	# pdb.set_trace()
plt.figure()
plt.plot(center_x_norm, center_y_norm, 'b.-')
plt.plot(center_x_norm[0], center_y_norm[0], 'rx')
plt.plot(center_x_norm[-1], center_y_norm[-1], 'gx')
plt.show()

# pdb.set_trace()
# dic = {'distance': dist_norm, 'center_x': center_x_norm, 'center_y':center_y_norm}
# pd.DataFrame(dic).to_csv('{}_norm.csv'.format(mapName))

## 저장 후, 너무 거리가 작은 것들은 손으로 마지막 행 삭제
# mugello, spa


# for i in range(len(dist)-1):
# 	ax, ay = inner_x[i], inner_y[i]
# 	bx, by = outer_x[i], outer_y[i]
# 	dist_ab = dist[i+1] - dist[i]



#	pass

# pdb.set_trace()

# imola: anomaly 4.xx (1)
