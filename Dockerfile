FROM python:3.12-slim
WORKDIR /opt/qtrace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . && python -m pytest -q
# reproduce every simulated result and figure
CMD ["make", "reproduce"]
