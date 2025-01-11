ARG PYTHON_VERSION=3.11.9
ARG AWS_CLI_VERSION=2.22.26

FROM amazon/aws-cli:${AWS_CLI_VERSION} AS aws-cli
FROM python:${PYTHON_VERSION}-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion \
        fonts-roboto \
        git-lfs \
        groff \
        jq \
        less \
        nano \
        rsync \
        zsh \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

WORKDIR /usr/local/share/fonts
RUN wget -qO- https://github.com/googlefonts/Montserrat/archive/refs/tags/v8.000.tar.gz | tar -xzvf - \
 && mv Montserrat-8.000/fonts/ttf/*.ttf ./ \
 && rm -rf Montserrat-8.000

COPY --from=aws-cli /usr/local/aws-cli/v2/current /usr/local

ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version=1.8.3
ENV PATH=${PATH}:${POETRY_HOME}/bin

WORKDIR /source
COPY pyproject.toml poetry.lock ./
RUN POETRY_VIRTUALENVS_CREATE=false \
    poetry install \
    --no-cache \
    --no-root \
    --with dev

ARG APP_USER=metr
ARG APP_DIR=/home/${APP_USER}/app
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} ${APP_USER} \
 && useradd -m -u ${USER_ID} -g ${APP_USER} -s /bin/bash ${APP_USER} \
 && mkdir -p ${APP_DIR}/.dvc/cache /home/${APP_USER}/.config/viv-cli \
 && chown -R ${USER_ID}:${GROUP_ID} ${APP_DIR} /home/${APP_USER}

WORKDIR ${APP_DIR}
COPY . .
RUN mkdir -p src \
 && touch src/__init__.py \
 && POETRY_VIRTUALENVS_CREATE=false \
    poetry install \
    --no-cache
USER ${APP_USER}
