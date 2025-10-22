# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 12:07:06 2025

@author: matil
"""
Liste et documentation des codes pour le travail de thèse de Martina :dizzy:


- interdistance



- densité référentielle

-- density : 
calcul de la densité référentielle
prends en input le fichier csv contenant chaque mention et ses caracteristiques
elimine du decompte les mentions imbriquées dans des mentions plus larges 
donne en sortie un tableau contenant la densité de chaque texte en plus du nb de tokens et le nombre de mentions 
utilisées pour le décompte

-- t_student_density:
applique le test t de student aux caracteristiques dispos (niveau, langue)
donne en sortie des boxplots pour répresenter les différentes combinaisons possibles et compare


- stabilité



- distribution des mentions par type de mention

mention_type_alternative et mention_type_alternative_ita
prends en entrée le csv contenant une mention par ligne
l'algo à base de règle établi quelle mention appartient à quel type, on obtient en sortie un fichier 
excel contenant une feuille pour type de mention
ATTENTION les fichiers excel ont été corrigés manuellement pour appliquer des statistiques dessus car il y avait des erreurs

type_mentions_plot --> plot sur types mentions

chains_patterns --> permet de créer un scatterplot par texte qui décrit une chaine par ligne et 
qui contient la succession des mentions par type (un symbole différent par type de mention) 
