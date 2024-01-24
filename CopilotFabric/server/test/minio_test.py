from minio.commonconfig import SnowballObject

from CopilotFabric.server.service import minio_client


def upload_model():
    minio_client.upload_snowball_objects(
        "copilotfabric",
        [
            SnowballObject(f"test_bpnn_avg_latency",
                           filename='F:/Project/PythonProject/Auto-Tuning-HLF/Model/bpnn/bpnn_avg_latency.pth'),
        ],
    )


if __name__ == '__main__':
    upload_model()
