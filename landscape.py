import pygame, sys
import math
import numba
from PIL import Image
from pygame.locals import *

def palette_swap(surf, old_c, new_c, image):
        img_copy = pygame.Surface(image.get_size())
        img_copy.fill(new_c)
        surf.set_colorkey(old_c)
        img_copy.blit(surf, (0, 0))
        return img_copy
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


def recalculate_data(map_data, size):
    """
    Function recalculates data
    """
    minimum, maximum = min(map_data), max(map_data)
    delta = maximum-minimum
    for i in range(size*size):
        map_data[i]=(map_data[i]-minimum)/delta
        map_data[i]*=map_data[i]
        # map_data[i]*=map_data[i]
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

def convert_landscape_map_to_image(map_data, size, name, rivers_map=None):
    """
    Function converts Perlin noise (landscape) to image
    """
    img = Image.new ("RGB", (size, size), (0, 0, 0))
    new_image = [0]*(size*size)
    for i in range(size*size):
        new_image[i] = set_color(map_data[0][i], int(max(WATER_LVL, map_data[0][i])*max_height))
    img.putdata(new_image)
    img.save(name)

def create_landscape_map(seed, size, size_of_side, number_of_octaves, rivers = False):
    """
    Creates perlin noise of landscape
    """
    map = create_Perlin_noise(seed, size, size_of_side, number_of_octaves)
    if rivers:
        seed = seed[::-1]
        for i in range(1000):
            seed=pseudo_random_generator(seed)
        rivers_map = create_Perlin_noise(seed, size, size_of_side, number_of_octaves)
        return (map, rivers_map)
    return (map, None)
def set_color(h, k):
    grass_rgb = [(3, 255, 56),(0, 230, 51),(0, 206, 45),(0, 183, 40),(0, 159, 34),(0, 137, 29),(0, 115, 23),(0, 94, 18),(0, 73, 12),(1, 54, 6)]
    water_rgb = [(0, 46, 62),(1, 60, 78),(2, 75, 95),(3, 90, 111),(4, 106, 128),(5, 122, 145),(7, 139, 162),(11, 156, 178),(16, 173, 195),(23, 191, 211)]
    mountain_rgb = [(124, 115, 104),(110, 107, 95),(97, 98, 88),(84, 90, 81),(73, 81, 75),(63, 72, 69),(55, 63, 62),(47, 54, 55),(40, 46, 47), (34, 37, 39)]
    snow_rgb = [(195, 252, 246),(202, 252, 247),(209, 253, 248),(216, 253, 249),(223, 254, 250),(229, 254, 251),(236, 254, 252),(242, 254, 253),(249, 255, 254),(254, 254, 254)]
    if h<=WATER_LVL:
        return water_rgb[int(h/WATER_LVL*10)%10 + int(h/WATER_LVL*10)//10*9]
    elif h<=SAND_LVL and k==int(max(WATER_LVL, h)*max_height):
        return (255,250,205)
    elif h<=SAND_LVL:
        return (255,250,205)
    elif h<=GRASS_LVL and k==int(h*max_height):
        return grass_rgb[(int((h-SAND_LVL)/(GRASS_LVL-SAND_LVL)*10))%10 + int((h-SAND_LVL)/(GRASS_LVL-SAND_LVL)*10)//10*9]
    elif h<=GRASS_LVL:
        return (139,69,19)
    elif h>MOUNTAIN_LVL and k==int(h*max_height):
        return snow_rgb[(int((h-MOUNTAIN_LVL)/(1-MOUNTAIN_LVL)*10))%10 + int((h-MOUNTAIN_LVL)/(1-MOUNTAIN_LVL)*10)//10*9]
    return mountain_rgb[(int((h-GRASS_LVL)/(1-GRASS_LVL)*10))%10 + int((h-GRASS_LVL)/(1-GRASS_LVL)*10)//10*9]

def make_darker(color, delta):
    return (max(color[0]-delta, 0), max(color[1]-delta, 0), max(color[2]-delta, 0))
def draw_image(arrays, name):
        screensize = size*width
        pygame.init()
        screen = pygame.display.set_mode((screensize, screensize), 0, 32)
        display = pygame.Surface((screensize, screensize))
        img=pygame.image.load(f'tiles/{height}x{width}.png').convert()
        array = arrays[0]
        river = array[1]
        img.set_colorkey((255, 255, 255))
        display.fill((255, 255, 255))
        for i in range(len(array)):
            x=i%size
            y=i//size
            if x != size-1 and y!=size-1:
                k = int(max(WATER_LVL, min(array[i+1], array[i], array[i+size]))*max_height)
            else:
                k=1
            while k<=int(max(WATER_LVL, array[i])*max_height):
                    color = set_color(array[i], k)
                    tile_img = palette_swap(img, (0, 255, 42), color, img)
                    tile_img = palette_swap(tile_img, (0, 0, 0), make_darker(color, 15), img)
                    tile_img = palette_swap(tile_img, (255, 0, 0), make_darker(color, 5), img)
                    tile_img.set_colorkey((255, 255, 255))
                    display.blit(tile_img, ((x-y)*width/2-width/2+screensize/2, height/2*(x+y)+screensize/2-k+1))
                    k+=1
        screen.blit(pygame.transform.scale(display, screen.get_size()), (0,0))
        pygame.display.update()
        pygame.image.save(screen, name)
#--------------------------------
WATER_LVL = 0.125
SAND_LVL = 0.15625
GRASS_LVL = 0.5625
MOUNTAIN_LVL = 0.6875
#--------------------------------
height = 2
width = 4
#--------------------------------
size = 256
size_of_side = 128
max_height = 256
octava=5
seed = "11211234"
#--------------------------------
map = create_landscape_map(seed, size, size_of_side, octava, False)
draw_image(map, "landscape141.png")
convert_landscape_map_to_image(map, size, "landscape17.png")