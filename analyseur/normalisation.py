import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

activities = ['walking', 'no_person', 'sitting', 'get_up', 'standing', 'get_down', 'lying']

main_directory = 'C:/Users/oussa/Downloads'

all_data = []
all_labels = []

for room in os.listdir(main_directory):
    room_path = os.path.join(main_directory, room)
    
    if os.path.isdir(room_path):
        for subdir in os.listdir(room_path):
            subdir_path = os.path.join(room_path, subdir)
            
            if os.path.isdir(subdir_path):
                label_file = os.path.join(subdir_path, 'label.csv')
                data_file = os.path.join(subdir_path, 'data.csv')
                
                if os.path.exists(label_file) and os.path.exists(data_file):
                    labels = pd.read_csv(label_file, header=None)
                    data = pd.read_csv(data_file, header=None)

                    print(f"Contenu de {label_file} :\n{labels.head()}")
                    
                    unique_activities = labels[1].unique()
                    print(f"Activités uniques dans {label_file} : {unique_activities}")

                    column_index = 1
                    
                    
                    for activity in activities:
                        activity_rows = labels[labels[column_index] == activity].index.tolist()
                        selected_indices = activity_rows[:30]
                        
                        for idx in selected_indices:
                            if idx < len(data):
                                all_data.append(data.iloc[idx].values)
                                all_labels.append(labels.iloc[idx, column_index])

if all_data:
    final_data = pd.DataFrame(all_data)
    final_labels = pd.Series(all_labels)

    final_data['label'] = final_labels.values

    scaler = StandardScaler()
    final_data_normalized = scaler.fit_transform(final_data.iloc[:, :-1])  # Normaliser seulement les données

    final_df = pd.DataFrame(final_data_normalized)
    final_df['label'] = final_labels.values

    final_df.to_csv('normalized_dataset.csv', index=False)

    print("Le fichier normalized_dataset.csv a été créé avec succès.")
else:
    print("Aucune donnée n'a été collectée. Vérifiez les activités dans les fichiers label.csv.")