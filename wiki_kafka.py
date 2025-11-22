import json
from confluent_kafka import Producer
import sseclient
import requests

# 1. Configuration Kafka
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)
topic = 'quickstart-events'

def delivery_report(err, msg):
    if err is not None:
        print(f' Erreur Kafka: {err}')
    # On commente le succès pour ne pas spammer la console
    # else:
    #     print(f'Message envoyé à {msg.topic()}')

print(" Connexion à Wikimedia...")

# 2. Connexion HTTP avec User-Agent (OBLIGATOIRE pour Wikipédia)
url = 'https://stream.wikimedia.org/v2/stream/recentchange'
headers = {'User-Agent': 'MonLaboDataMaster/1.0 (contact@example.com)'}

try:
    response = requests.get(url, stream=True, headers=headers)
    # Vérifie si Wikipédia nous a jeté (ex: erreur 403 ou 404)
    if response.status_code != 200:
        print(f" Erreur de connexion au flux : Code {response.status_code}")
        print(response.text)
        exit()
        
    client = sseclient.SSEClient(response)
    print(" Connecté ! En attente d'événements...")

    for event in client.events():
        if event.event == 'message':
            try:
                data = json.loads(event.data)
                
                # Extraction des infos intéressantes
                user = data.get('user', 'inconnu')
                title = data.get('title', 'sans titre')
                wiki = data.get('server_name', 'wiki')
                
                # Création du message
                msg_value = f"[{wiki}] {user} a modifié : {title}"
                
                # Affichage dans la console (pour que tu voies que ça marche)
                print(f"📨 {msg_value}")

                # Envoi vers Kafka
                producer.produce(topic, value=msg_value.encode('utf-8'), callback=delivery_report)
                
                # Important : poll permet de gérer les envois
                producer.poll(0)
                
            except Exception as e:
                print(f"Erreur de traitement : {e}")

except KeyboardInterrupt:
    print("\ Arrêt demandé par l'utilisateur.")
except Exception as global_e:
    print(f"\ Erreur critique : {global_e}")
finally:
    print("Envoi des derniers messages...")
    producer.flush()
    print("Terminé.")