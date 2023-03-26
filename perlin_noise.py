"""
Module generates Perman noise
"""
import math
import numba
from PIL import Image
from flask import Flask, render_template
import shutil


def pseudo_random_generator(seed:str) -> str:
    """
    Function generates pseudo random numbers
    """
    seed_num=int(seed)
    seed=str((seed_num*seed_num+31513)%100000000)
    return '0'*(8-len(seed))+seed

def get_number(seed:str) -> int:
    """
    Function returns angle for vector from seed
    """
    return (int(seed[int(seed[3])%8])*13 +int(seed[int(seed[7])%8])*3)%36*10

@numba.njit(fastmath=True)
def vector_miltiplication(vector, alpha):
    """
    Function returns vector multiplication (second vector is )
    """
    return vector[0]*math.cos(alpha)+vector[1]*math.sin(alpha)

@numba.njit(fastmath=True)
def fill_one_pixel(y_pixel_vector, x_pixel_vector, vector):
    """
    Function fills one pixel due to data
    """
    return (vector_miltiplication((y_pixel_vector, x_pixel_vector), vector)+1)/2

def create_vectors_map(size, seed):
    """
    Function creates vectors array in spiral way
    """
    vector_array= [[0]*(size+1) for i in range(size+1)]
    x_coor, y_coor =size//2, size//2
    k=1
    for j in range(size//2):
        for i in range(k):
            vector_array[y_coor][x_coor+i]=get_number(seed)
            seed=pseudo_random_generator(seed)
        x_coor+=k
        for i in range(k):
            vector_array[y_coor-i][x_coor]=get_number(seed)
            seed=pseudo_random_generator(seed)
        y_coor-=k
        k+=1
        for i in range(k):
            vector_array[y_coor][x_coor-i]=get_number(seed)
            seed=pseudo_random_generator(seed)
        x_coor-=k
        for i in range(k):
            vector_array[y_coor+i][x_coor]=get_number(seed)
            seed=pseudo_random_generator(seed)
        y_coor+=k
        k+=1
    for i in range(size-1):
        vector_array[y_coor][x_coor+i]=get_number(seed)
        seed=pseudo_random_generator(seed)
    return vector_array

def convert_to_image(map_data, size, name, colors=None, start_color = (0, 0, 0), finish_color = (255, 255, 255)):
    """
    Function converts data to image
    """
    img = Image.new ("RGB", (size, size), (0, 0, 0))
    new_image = [0]*(size*size)
    if colors != None:
        first_element = list(colors.keys())[0]
    for i in range(size*size):
        if colors!=None:
            current_color = colors[first_element]
            for k in colors:
                if map_data[i]<=k:
                    current_color = colors[k]
                    break
        else:
            current_color = (int(start_color[0]+(finish_color[0]-start_color[0])*map_data[i]), int(start_color[1]+(finish_color[1]-start_color[1])*map_data[i]), int(start_color[2]+(finish_color[2]-start_color[2])*map_data[i]))
        new_image[i]=current_color
    img.putdata(new_image)
    img.save(name)

def recalculate_data(map_data, size):
    """
    Function recalculates data
    """
    minimum, maximum = min(map_data), max(map_data)
    delta = maximum-minimum
    for i in range(size*size):
        map_data[i]=(map_data[i]-minimum)/delta
    return map_data

@numba.njit(fastmath=True)
def smoothing(pixel1, pixel2, i, size_of_side):
    """
    Function smoothes point (linear interpolation + smooth stepping)
    """

    t = (i % size_of_side) / size_of_side
    return pixel1 + (pixel2 - pixel1) * t*t*t*(t*(t*6-15)+10)

def create_octava_n(size, seed, size_of_side):
    """
    Function creates n`octava Perlin noise
    """
    octava=size//size_of_side
    vectors_map=create_vectors_map(octava, seed)
    map = [0]*(size*size)
    for i in range(size):
        pixel_y_dist = i%size_of_side/size_of_side
        y_size = i//size_of_side
        for j in range(size):
            pixel_x_dist = j%size_of_side/size_of_side
            x_size = j//size_of_side
            # l = left, r = right, u = up, d = down
            pixel_l_u = fill_one_pixel(-pixel_y_dist, -pixel_x_dist, vectors_map[y_size][x_size])
            pixel_r_u = fill_one_pixel(-pixel_y_dist, 1-pixel_x_dist, vectors_map[y_size][x_size+1])
            pixel_l_d = fill_one_pixel(1-pixel_y_dist, -pixel_x_dist, vectors_map[y_size+1][x_size])
            pixel_r_d = fill_one_pixel(1-pixel_y_dist, 1-pixel_x_dist, vectors_map[y_size+1][x_size+1])
            pixel_u = smoothing(pixel_l_u, pixel_r_u, j, size_of_side)
            pixel_d = smoothing(pixel_l_d, pixel_r_d, j, size_of_side)
            map[i*size+j] = smoothing(pixel_u, pixel_d, i, size_of_side)
    return map

def sum_all_maps(size, array_of_maps):
    """
    Function sums every layer of the Perlin noise
    """
    for i in range(size*size):
        for k in range(1, len(array_of_maps)):
            array_of_maps[0][i]+=array_of_maps[k][i]/(1<<k)
    return recalculate_data(array_of_maps[0], size)

def create_Perlin_noise(seed, size, size_of_side, number_of_octaves):
    """
    Function creates Perlin noise
    """
    array_of_maps=[0]*number_of_octaves
    for i in range(number_of_octaves):
        array_of_maps[i] = create_octava_n(size, seed, size_of_side)
        size_of_side>>=1
    return sum_all_maps(size, array_of_maps)
########

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("main.html")

@app.route('/result', methods=['POST'])
def result():
    size=64
    seed="31941951"
    octaves=5
    size_of_side=32
    perlin_noise=create_Perlin_noise(seed, size, size_of_side, octaves)
    colors = {0.1:(3, 255, 56),0.2:(0, 230, 51),
              0.3:(0, 206, 45),0.4:(0, 183, 40),
              0.5:(0, 159, 34),0.6:(0, 137, 29),
              0.7:(0, 115, 23),0.8:(0, 94, 18),
              0.9:(0, 73, 12),1:(1, 54, 6)}

    convert_to_image(perlin_noise, size, "perlin36.png", colors)

    file_path = 'perlin36.png'
    dst_folder = 'static/perlin36.png'
    shutil.move(file_path, dst_folder)

    # Render the result page with the output
    return render_template('result.html')

if __name__ == "__main__":

    app.run(debug=True)
