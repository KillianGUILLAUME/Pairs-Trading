"""
neural_nets/utils/launcher.py

Lance (optionnellement) la construction du dataset causal puis l'entraînement de la
Neural SDE sur une instance Spot EC2, rapatrie les artefacts, détruit l'instance.

    venv/bin/python neural_nets/utils/launcher.py                   # entraînement seul
    venv/bin/python neural_nets/utils/launcher.py --build-dataset   # screener (toutes les paires) puis entraînement
    venv/bin/python neural_nets/utils/launcher.py --dataset-only    # screener seul
    venv/bin/python neural_nets/utils/launcher.py --dry-run         # affiche les commandes sans rien lancer

Variables `.env` : AWS_REGION, AMI_ID, INSTANCE_TYPE, KEY_NAME, PEM_KEY_PATH,
WANDB_API_KEY (optionnel), LOCAL_DOWNLOAD_PATH (optionnel, défaut ./data),
DATA_BUCKET (optionnel : registre S3, voir scripts/data_registry.py).

Ce qui est envoyé : le code + data/storage/parquet (données brutes 1h). Ce qui est
rapatrié : data/models/*.pt (toutes les 5 min et à la fin) et le dataset causal
(super_dataset_*_causal.parquet + _meta.json) pour qu'il soit enregistré localement.
"""

import argparse
import os
import subprocess
import threading
import time

import boto3
from dotenv import load_dotenv
from fabric import Connection

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AMI_ID = os.getenv("AMI_ID")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")
KEY_NAME = os.getenv("KEY_NAME")
KEY_PATH = os.getenv("PEM_KEY_PATH")

LOCAL_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_DOWNLOAD_PATH = os.getenv("LOCAL_DOWNLOAD_PATH") or os.path.join(LOCAL_PROJECT_PATH, "data")
REMOTE_DIR = "/home/ubuntu/pairs_trading"

SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Tout ce qui n'est pas nécessaire au calcul reste local (temps de transfert et disque)
RSYNC_EXCLUDES = [
    "venv", ".venv", ".git", "__pycache__", ".env", ".DS_Store", "*.pyc",
    "data/models",                      # checkpoints : jamais envoyés, seulement rapatriés
    "data/storage/screened/raw_pairs*", "data/storage/screened/screened_*", "data/storage/screened/multi_window*",
    "data/storage/screened/super_dataset_SDE_128.parquet",   # ancien dataset non causal
    "data/storage/synthetic_paths", "data/storage/massive_synthetic_paths.parquet",
    "data/storage/parquet/4h", "data/storage/parquet/1d",   # seul le 1h sert au screener / SDE
    "notebooks", "docs", "legacy", "tests", "*.docx", "*.ipynb", "wandb",
]

REMOTE_DATASET_GLOB = "data/storage/screened/super_dataset_*_causal*"


def _ssh(key_path):
    return f"ssh -i {key_path} {SSH_OPTS}"


def rsync_pull(ip, key_path, remote_subdir, local_dir, includes=None, quiet=True):
    """Rapatrie `remote_subdir` (relatif à REMOTE_DIR) dans `local_dir`, sans écraser l'existant."""
    os.makedirs(local_dir, exist_ok=True)
    cmd = ["rsync", "-az", "--ignore-existing", "-e", _ssh(key_path)]
    if includes:
        cmd += ["--include=*/"] + [f"--include={p}" for p in includes] + ["--exclude=*"]
    cmd += [f"ubuntu@{ip}:{REMOTE_DIR}/{remote_subdir}/", local_dir]
    kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
    subprocess.run(cmd, timeout=600, **kw)


def background_sync_task(ip_address, key_path, local_path, stop_event):
    print("🔄 [AUTO-SYNC] Activé (toutes les 5 min : data/models -> local)...")
    while not stop_event.is_set():
        try:
            rsync_pull(ip_address, key_path, "data/models", os.path.join(local_path, "models"))
        except Exception:
            pass
        if stop_event.wait(300):
            break


def build_remote_commands(build_dataset: bool, dataset_only: bool, n_jobs: int | None):
    cmds = []
    if build_dataset or dataset_only:
        jobs = f" --n_jobs {n_jobs}" if n_jobs else ""
        cmds.append(f"python research/pairs/build_screened_dataset.py{jobs}")
    if not dataset_only:
        cmds.append("python neural_nets/train.py")
    return cmds


def main():
    parser = argparse.ArgumentParser(description="Dataset causal + entraînement Neural SDE sur AWS EC2 Spot")
    parser.add_argument("--build-dataset", action="store_true", help="Construire le dataset causal avant d'entraîner")
    parser.add_argument("--dataset-only", action="store_true", help="Construire le dataset causal, sans entraîner")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallélisme du screener sur l'instance")
    parser.add_argument("--instance-type", default=None, help="Surcharge INSTANCE_TYPE du .env (ex. c6i.4xlarge pour le dataset)")
    parser.add_argument("--volume-gb", type=int, default=80)  # l'AMI exige >= 75 Go
    parser.add_argument("--dry-run", action="store_true", help="Afficher les commandes sans lancer d'instance")
    args = parser.parse_args()

    instance_type = args.instance_type or INSTANCE_TYPE
    remote_cmds = build_remote_commands(args.build_dataset, args.dataset_only, args.n_jobs)
    rsync_push = ["rsync", "-az", "--delete", "-e", f"ssh -i {KEY_PATH} {SSH_OPTS}"]
    for ex in RSYNC_EXCLUDES:
        rsync_push += ["--exclude", ex]
    rsync_push += [os.path.join(LOCAL_PROJECT_PATH, ""), f"ubuntu@<IP>:{REMOTE_DIR}"]

    if args.dry_run:
        print("Commandes distantes :", *[f"  cd {REMOTE_DIR} && source venv/bin/activate && {c}" for c in remote_cmds], sep="\n")
        print("Rsync :", " ".join(rsync_push))
        return

    if not os.path.exists(os.path.join(LOCAL_PROJECT_PATH, "requirements.txt")):
        print("❌ ERREUR : pas de requirements.txt à la racine du projet.")
        return
    for name, val in (("AWS_REGION", AWS_REGION), ("AMI_ID", AMI_ID), ("INSTANCE_TYPE", instance_type),
                      ("KEY_NAME", KEY_NAME), ("PEM_KEY_PATH", KEY_PATH)):
        if not val:
            print(f"❌ ERREUR : variable {name} absente du .env")
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
        except Exception:
            vpc_id = ec2_client.describe_vpcs()['Vpcs'][0]['VpcId']
            sg = ec2_client.create_security_group(GroupName="Pairs-Trading-SSH-Access", Description="SSH", VpcId=vpc_id)
            sg_id = sg['GroupId']
            ec2_client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[
                {'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}])

        print(f"🚀 Lancement sur {instance_type}...")
        instances = ec2.create_instances(
            ImageId=AMI_ID, InstanceType=instance_type, KeyName=KEY_NAME, MinCount=1, MaxCount=1,
            BlockDeviceMappings=[{'DeviceName': '/dev/sda1', 'Ebs': {'VolumeSize': args.volume_gb, 'VolumeType': 'gp3', 'DeleteOnTermination': True}}],
            InstanceMarketOptions={'MarketType': 'spot', 'SpotOptions': {'SpotInstanceType': 'one-time'}},
            SecurityGroupIds=[sg_id],
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'Pairs-Trading-NeuralSDE'}]}]
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
                conn.run("echo ok", hide=True, timeout=5)
                break
            except Exception:
                time.sleep(5)
        else:
            print("⚠️ SSH indisponible après 2.5 min, on tente rsync quand même...")

        print("📂 Envoi du code + données brutes 1h...")
        rsync_push[-1] = f"ubuntu@{ip_address}:{REMOTE_DIR}"
        subprocess.run(rsync_push, stdout=subprocess.DEVNULL, check=True)

        print("🛠️ Installation...")
        conn.run("sudo apt update -qq && sudo apt install -y python3-pip", hide=True, timeout=300)
        conn.run("sudo pip3 install virtualenv", hide=True, timeout=60)
        conn.run(f"cd {REMOTE_DIR} && virtualenv venv", timeout=60)
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --no-cache-dir -q 'setuptools<70.0.0' 'numpy<2' wheel", timeout=300)
        if not args.dataset_only:
            conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --no-cache-dir -q torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118", timeout=600)
        req_file = "requirements.txt"
        if args.dataset_only:   # phase CPU : ni torch ni torchsde
            conn.run(f"cd {REMOTE_DIR} && grep -v -E '^(torch|torchsde)' requirements.txt > requirements_cpu.txt")
            req_file = "requirements_cpu.txt"
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --no-cache-dir -q -r {req_file}", timeout=900)

        sync_thread = threading.Thread(target=background_sync_task, args=(ip_address, KEY_PATH, LOCAL_DOWNLOAD_PATH, stop_sync_event), daemon=True)
        sync_thread.start()

        # Secrets transmis par l'environnement de la session SSH, jamais dans la ligne de commande (visible dans `ps`)
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        if os.getenv("WANDB_API_KEY"):
            env["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
        for cmd in remote_cmds:
            print(f"🔥 {cmd}")
            conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && {cmd}", pty=True, env=env)

    except KeyboardInterrupt:
        print("\n🛑 ARRÊT MANUEL (Ctrl+C)")
    except Exception as e:
        print(f"❌ ERREUR : {e}")
    finally:
        print("\n🧹 Nettoyage...")
        stop_sync_event.set()
        if ip_address:
            try:
                rsync_pull(ip_address, KEY_PATH, "data/models", os.path.join(LOCAL_DOWNLOAD_PATH, "models"))
                rsync_pull(ip_address, KEY_PATH, "data/storage/screened", os.path.join(LOCAL_DOWNLOAD_PATH, "storage", "screened"),
                           includes=["super_dataset_*_causal*"])
                print("📥 Checkpoints et dataset causal rapatriés.")
            except Exception as e:
                print(f"⚠️ Rapatriement incomplet : {e}")
        if instance:
            print("💀 Destruction instance...")
            instance.terminate()
            print("✅ Terminée.")


if __name__ == "__main__":
    main()
