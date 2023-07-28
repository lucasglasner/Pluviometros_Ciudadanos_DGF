#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: Fri Apr 29 09:05:46 2022
 # @ Modified by: 
 # @ Modified time: 2023-07-28 09:53:15
 # @ Description:
 '''

# =============================================================================
# IMPORTAR LIBRERIAS Y COSAS GENERALES
# =============================================================================
import numpy as np                     # <- Calculos numericos
import pandas as pd                    # <- Tablas
import matplotlib.pyplot as plt        # <- Graficos
import matplotlib as mpl
import cartopy.crs as ccrs             # <- Mapas y cartografia
import cartopy.io.img_tiles            
from utils import scale_bar

title   = "Evento 26 - 27 abril 2022"                                # <- Titulo del grafico
fformat = ".pdf"                                                     # <- Formato de salida de las figuras (pdf, jpg, png, etc)
spreadsheet = "planillas/Evento 26 - 27 abril 2022.xlsx"             # <- Ruta a la planilla con los datos
sheetname   = "Resumen"                                              # <- Nombre de la hoja que tiene los datos en el excel

# =============================================================================
# LEER DATOS EN LA PLANILLA EXCEL
# =============================================================================

print('Leyendo datos de la planilla...')
datos = pd.read_excel(spreadsheet, sheet_name=sheetname)             # <- Nombre de la hoja con los datos
print(datos,'\n')
print('Corriendo un simple "control de calidad"...')
#Corregir algunos detalles, cambiar comas por puntos y asegurarse que las 
#coordenadas son "negativas". 
for cord in ["lon","lat"]:
    datos.loc[:,cord] = list(map(lambda x: float(str(x).replace(",",".")),
                                 datos.loc[:,cord]))
    datos.loc[:,cord] = list(map(lambda x: -x if x>0 else x,
                                 datos.loc[:,cord]))


sindatos = datos[np.isnan(datos['pp'].values)]             # <- Estacion sin info de lluvia
datos = datos[~np.isnan(datos["pp"]).values]               # <- Dejar solo estaciones con info de lluvia


# =============================================================================
# CARGAR IMAGENES ADICIONALES A LA FIGURA
# =============================================================================
print('Cargando imagenes extras (logos dgf, ppcc, google earth, etc)\n')
request = cartopy.io.img_tiles.GoogleTiles(style="satellite")  # <- Solicitar imagen de google earth de fondo
leyenda = plt.imread('static/leyenda2.png')                    # <- Ruta a la imagen con la leyenda
logoPC  = plt.imread("static/logo_pc.png")                     # <- Ruta a la imagen con el logo de ppcc
logodgf = plt.imread("static/dgf.png")                         # <- Ruta a la imagen con el logo del dgf

# =============================================================================
# GRAFICO DOMINIO COMPLETO
# =============================================================================
print('Graficando precipitacion en el dominio de la RM')
#Construir la figura y los ejes
fig = plt.figure(figsize=(10,10),num=0)
ax  = fig.add_subplot(111,projection=ccrs.PlateCarree())
ax.set_extent([-71.47, -70.03, -34.05, -32.85])  # <- Extension geográfica de la figura: lon1,lon2,lat1,lat2
ax.gridlines(linestyle=":")

#Definir escala y agrupar los datos segun ésta.
bins    = np.arange(0,50+5,5)    # <- Limites: np.arange(0, n+k, k) Desde 0 hasta n cada k milimetros.
bins   = np.hstack((bins,float("inf")))
binned = (pd.cut(datos["pp"],bins,right=False))
grouped = datos.groupby(binned)

#Definir colores, etiquetas de la leyenda y tamaños de las pelotitas.
colores = mpl.cm.YlGnBu(np.linspace(0,1,len(bins))) # <- Definir paleta de colores. Otros colores acá: https://matplotlib.org/stable/tutorials/colors/colormaps.html
sizes = sorted(colores.mean(axis=1)*0+colores.mean(axis=1)**3*1000)
labels  = np.empty(len(bins)-1,dtype="U100")
for i in range(len(bins)-2):
    labels[i] = "["+"{:.0f}".format(bins[i])+","+"{:.0f}".format(bins[i+1])+")"
labels[len(bins)-2] = "["+"{:.0f}".format(bins[len(bins)-2])+",)"

#Poner puntos en el mapa segun su tamaño y color.
for i, (name,group) in enumerate(grouped):
    mask = group["grupo"] == "Estación" # <- Mascara para diferenciar estaciones de ppcc
    #Graficar estaciones con un achurado
    ax.scatter(group[mask].lon,group[mask].lat,s=sizes[i],hatch="xxxx",alpha=0.8,
               color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())
    #Graficar ppcc sin achurado
    ax.scatter(group[~mask].lon,group[~mask].lat,s=sizes[i],alpha=0.8,label=labels[i],
               color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())

scale_bar(ax=ax,length=20,location=(0.1,0.03), fs=18, col="w")  # <- Colocar barra de escala de 20km
ax.add_image(request,11)                                        # <- Colocar imagen google earth de fondo
ax.legend(loc=(1.1,0),title="Precipitación (mm)",frameon=False) # <- Colocar leyenda
ax.set_title(title,fontsize=18, loc='left')                     # <- Colocar titulo
ax.set_aspect('auto')                                           

# Poner logo de ppcc
ax1 = fig.add_axes([ax.get_position().xmax+0.05,0.75,0.18,0.18])
ax1.axis("off")
ax1.imshow(logoPC)

# Poner logo del dgf
ax2 = fig.add_axes([ax.get_position().xmax+0.05,0.65,0.18,0.18])
ax2.axis("off")
ax2.imshow(logodgf)

# Poner leyenda que diferencia estaciones de ppcc
ax3 = fig.add_axes([ax.get_position().xmax+0.05,0.4,0.13,0.13])
ax3.axis("off")
ax3.imshow(leyenda)

print('Guardando pdf...\n')
# Guardar figura como pdf
plt.savefig("plots/Tormenta1_"+title+fformat,dpi=150,bbox_inches="tight")



# =============================================================================
# GRAFICO ZOOM EN SANTIAGO
# =============================================================================
print('Graficando precipitacion en el dominio de Santiago...')
#Construir la figura y los ejes
fig = plt.figure(figsize=(10,10),num=1)
ax  = fig.add_subplot(111,projection=ccrs.PlateCarree())
ax.set_extent([-70.87, -70.43, -33.63, -33.27])  # <- Extension geográfica de la figura: lon1,lon2,lat1,lat2
ax.gridlines(linestyle=":")


#Definir escala y agrupar los datos segun ésta.
bins    = np.arange(0,50+5,5)    # <- Limites: np.arange(0, n+k, k) Desde 0 hasta n cada k milimetros.
bins   = np.hstack((bins,float("inf")))
binned = (pd.cut(datos["pp"],bins,right=False))
grouped = datos.groupby(binned)

#Definir colores, etiquetas de la leyenda y tamaños de las pelotitas.
colores = mpl.cm.YlGnBu(np.linspace(0,1,len(bins))) # <- Definir paleta de colores. Otros colores acá: https://matplotlib.org/stable/tutorials/colors/colormaps.html
sizes = sorted(colores.mean(axis=1)*0+colores.mean(axis=1)**3*1000)
labels  = np.empty(len(bins)-1,dtype="U100")
for i in range(len(bins)-2):
    labels[i] = "["+"{:.0f}".format(bins[i])+","+"{:.0f}".format(bins[i+1])+")"
labels[len(bins)-2] = "["+"{:.0f}".format(bins[len(bins)-2])+",)"

#Poner puntos en el mapa segun su tamaño y color.
for i, (name,group) in enumerate(grouped):
    mask = group["grupo"] == "Estación" # <- Mascara para diferenciar estaciones de ppcc
    #Graficar estaciones con un achurado
    ax.scatter(group[mask].lon,group[mask].lat,s=sizes[i],hatch="xxxx",alpha=0.8,
               color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())
    #Graficar ppcc sin achurado
    ax.scatter(group[~mask].lon,group[~mask].lat,s=sizes[i],alpha=0.8,label=labels[i],
               color=colores[i],edgecolor="k",transform=ccrs.PlateCarree())

scale_bar(ax=ax,length=20,location=(0.1,0.03), fs=18, col="w")  # <- Colocar barra de escala de 20km
ax.add_image(request,11)                                        # <- Colocar imagen google earth de fondo
ax.legend(loc=(1.1,0),title="Precipitación (mm)",frameon=False) # <- Colocar leyenda
ax.set_title(title,fontsize=18, loc='left')                     # <- Colocar titulo
ax.set_aspect('auto')                                           

# Poner logo de ppcc
ax1 = fig.add_axes([ax.get_position().xmax+0.05,0.75,0.18,0.18])
ax1.axis("off")
ax1.imshow(logoPC)

# Poner logo del dgf
ax2 = fig.add_axes([ax.get_position().xmax+0.05,0.65,0.18,0.18])
ax2.axis("off")
ax2.imshow(logodgf)

# Poner leyenda que diferencia estaciones de ppcc
ax3 = fig.add_axes([ax.get_position().xmax+0.05,0.4,0.13,0.13])
ax3.axis("off")
ax3.imshow(leyenda)

print('Guardando como pdf')
# Guardar figura como pdf
plt.savefig("plots/Tormenta2_"+title+fformat,dpi=150,bbox_inches="tight")
