import os
import io
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

class FolderAccess:
    def __init__(self):
        pass
    
    def get_all_file(self):
        pass
    def read_csv_by_data_frame(self, file_name: str) -> pd.DataFrame:
        pass

class AzureStorageAccess(FolderAccess):
    def __init__(self):
        super().__init__()
        load_dotenv()
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING 환경 변수가 설정되지 않았습니다.")
        self.blob_service = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.blob_service.get_container_client("data")

    def get_all_file(self):
        # 3) 모든 blob 목록 가져오기
        return self.container_client.list_blobs()

    def read_csv_by_data_frame(self, file_name: str) -> pd.DataFrame:
        try:
            # 3) Blob 다운로드 스트림 받기
            # 파일이 없으면 여기서 ResourceNotFoundError가 발생합니다.
            download_stream = self.container_client.download_blob(file_name)
            stream_bytes = download_stream.readall()  # bytes로 읽어옴

            # 4) pandas로 바로 읽기
            df = pd.read_csv(io.BytesIO(stream_bytes))
            return df

        except ResourceNotFoundError:
            # 파일이 존재하지 않을 경우 실행되는 부분
            print(f"Error: Blob '{file_name}' not found.")
            # 빈 DataFrame을 반환하여 프로그램이 중단되지 않도록 합니다.
            return pd.DataFrame()
        except Exception as e:
            # 그 외 다른 종류의 에러가 발생했을 경우
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()
    