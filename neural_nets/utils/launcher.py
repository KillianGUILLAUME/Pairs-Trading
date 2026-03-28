import boto3
import time
import os
import subprocess
import threading
from fabric import Connection
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AMI_ID = os.getenv("AMI_ID")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")
KEY_NAME = os.getenv("KEY_NAME")
KEY_PATH = os.getenv("PEM_KEY_PATH")

LOCAL_DOWNLOAD_PATH = os.getenv("LOCAL_DOWNLOAD_PATH") or os.path.join(os.getcwd(), "data")
LOCAL_PROJECT_PATH = os.getcwd()
REMOTE_DIR = "/home/ubuntu/pairs_trading"

def background_sync_task(ip_address, key_path, remote_dir, local_path, stop_event):
    """ 
    Sauvegarde silencieuse et intelligente des modèles génératifs depuis AWS.
    --ignore-existing : Si le fichier est déjà là, on ne le retélécharge PAS.
    """
    print(f"🔄 [AUTO-SYNC] Activé (Toutes les 5min)...")
    os.makedirs(os.path.join(local_path, "models"), exist_ok=True)

    while not stop_event.is_set():
        try:
            subprocess.run([
                "rsync", "-az", "--ignore-existing", 
                "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                f"ubuntu@{ip_address}:{remote_dir}/data/models/",
                os.path.join(local_path, "models")
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
        
        if stop_event.wait(300):
            break

import argparse

def main():
    parser = argparse.ArgumentParser(description="Lance l'entraînement du modèle génératif Pairs Trading sur AWS EC2")
    args = parser.parse_args()

    if not os.path.exists("requirements.txt"):
        print(f"❌ ERREUR : Pas de requirements.txt à la racine du projet.")
        return

    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    ec2 = boto3.resource('ec2', region_name=AWS_REGION)
    instance = None
    ip_address = None
    stop_sync_event = threading.Event()

    try:
        try:
            sg = ec2_client.describe_security_groups(GroupNames=["Pairs-Trading-SSH-Access"])['SecurityGroups'][0]
            sg_id = sg['GroupId']
        except:
            vpc_id = ec2_client.describe_vpcs()['Vpcs'][0]['VpcId']
            sg = ec2_client.create_security_group(GroupName="Pairs-Trading-SSH-Access", Description="SSH", VpcId=vpc_id)
            sg_id = sg['GroupId']
            ec2_client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[{'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}])
        
        print(f"🚀 Lancement sur {INSTANCE_TYPE}...")

        instances = ec2.create_instances(
            ImageId=AMI_ID, InstanceType=INSTANCE_TYPE, KeyName=KEY_NAME, MinCount=1, MaxCount=1,
            BlockDeviceMappings=[{'DeviceName': '/dev/sda1', 'Ebs': {'VolumeSize': 80, 'VolumeType': 'gp3', 'DeleteOnTermination': True}}],
            InstanceMarketOptions={'MarketType': 'spot', 'SpotOptions': {'SpotInstanceType': 'one-time'}}, 
            SecurityGroupIds=[sg_id],
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'Pairs-Trading-Bot-Prod'}]}]
        )
        instance = instances[0]
        instance.wait_until_running()
        instance.reload()
        ip_address = instance.public_ip_address
        print(f"✅ IP : {ip_address}")

        print("🔌 Attente SSH...")
        conn = Connection(host=ip_address, user="ubuntu", connect_kwargs={"key_filename": KEY_PATH})
        for _ in range(30):
            try:
                conn.run("echo 'SSH est prêt'", hide=True, timeout=5)
                break
            except Exception:
                time.sleep(5)
        else:
            print("⚠️ Impossible de se connecter en SSH après 2.5 minutes, on tente rsync quand même...")

        print("📂 Envoi du code...")
        subprocess.run([
            "rsync", "-az",
            "-e", f"ssh -i {KEY_PATH} -o StrictHostKeyChecking=no",
            "--exclude", "venv", "--exclude", ".venv", "--exclude", ".git", "--exclude", "__pycache__", "--exclude", ".env", "--exclude", ".DS_Store", "--exclude", "data/models",
            os.path.join(LOCAL_PROJECT_PATH, ""), 
            f"ubuntu@{ip_address}:{REMOTE_DIR}"
        ], stdout=subprocess.DEVNULL, check=True)

        print("🛠️ Installation silencieuse...")
        # L'image AWS DLAMI a un conflit de version interne sur le paquet python3-venv via apt.
        # On contourne le problème 'held broken packages' d'apt en utilisant la librairie python `virtualenv`.
        conn.run("sudo apt update -qq && sudo apt install -y python3-pip", hide=True, timeout=300)
        conn.run("sudo pip3 install virtualenv", hide=True, timeout=60)
        conn.run(f"cd {REMOTE_DIR} && virtualenv venv", timeout=60)
        
        # Installation des dépendances depuis requirements.txt
        # numpy est installé d'abord car iisignature a besoin de ses headers C pour compiler (ModuleNotFoundError numpy)
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --no-cache-dir -q 'setuptools<70.0.0' 'numpy<2' wheel", timeout=300)
        # torch est installe en amont pour preserver sa priorite (CUDA natif linux) et compiler signatory correctement
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --no-cache-dir -q torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118", timeout=600)
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --no-cache-dir -q --no-build-isolation -r requirements.txt", timeout=600)

        sync_thread = threading.Thread(
            target=background_sync_task,
            args=(ip_address, KEY_PATH, REMOTE_DIR, LOCAL_DOWNLOAD_PATH, stop_sync_event),
            daemon=True 
        )
        sync_thread.start()

        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            train_cmd = f"export WANDB_API_KEY={wandb_key} && python neural_nets/train.py"
        else:
            train_cmd = f"python neural_nets/train.py"

        print(f"🔥 Lancement Entraînement (neural_nets/train.py)...")
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && {train_cmd}", pty=True)

    except KeyboardInterrupt:
        print("\n🛑 ARRÊT MANUEL (Ctrl+C)")
        
    except Exception as e:
        print(f"❌ ERREUR : {e}")

    finally:
        print("\n🧹 Nettoyage...")
        if stop_sync_event: stop_sync_event.set()
        
        if ip_address:
            try:
                subprocess.run(["rsync", "-az", "--ignore-existing", "-e", f"ssh -i {KEY_PATH} -o StrictHostKeyChecking=no", f"ubuntu@{ip_address}:{REMOTE_DIR}/data/models/", os.path.join(LOCAL_DOWNLOAD_PATH, "models")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except: pass

        if instance:
            print("💀 Destruction instance...")
            instance.terminate()
            print("✅ Terminée.")

if __name__ == "__main__":
    main()