FROM node:20

WORKDIR /app

# Install dependencies first for better layer caching
COPY webapp/package*.json ./
RUN npm ci

# Bundle source so the image can run standalone; bind mounts will override in dev
COPY webapp ./

ENV HOST=0.0.0.0
EXPOSE 3000

CMD ["npm", "start"]
