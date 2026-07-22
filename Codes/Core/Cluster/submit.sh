#!/bin/bash
#$ -cwd            # Ejecuta en el directorio actual [cite: 123, 172]
#$ -N prueba    # Nombre del trabajo [cite: 125, 143]
#$ -j y             # Une la salida y el error en un solo archivo [cite: 127, 145]
#$ -S /bin/bash     # Usa bash como shell [cite: 129, 147]
#$ -l mem=1G        # Pide memoria RAM (obligatorio) [cite: 132, 150]
#$ -V               # Exporta las variables de entorno actuales [cite: 135, 173]

module load miniconda
# source activate mi_entorno  # Actívalo si usas un entorno propio [cite: 230]
python prueba.py