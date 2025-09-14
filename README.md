# Implementación del algoritmo A*
Este documento proporciona las instrucciones necesarias para ejecutar este proyecto.

## Requisitos
Para correr este proyecto, es recomendable instalar `uv`, un gestor de paquetes y proyectos de Python.

### Instalación de `uv`
Puedes encontrar más información sobre `uv` en la [documentación oficial](https://docs.astral.sh/uv/).

#### Windows
Abre una terminal de PowerShell y ejecuta el siguiente comando:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux
Abre una terminal y ejecuta el siguiente comando:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Ejecución del Proyecto
Una vez que `uv` esté instalado, puedes ejecutar la aplicación principal con el siguiente comando en tu terminal. `uv` se encargará de gestionar las dependencias y el entorno necesarios para ejecutar el script.

```shell
uv run app.py
```

### Alternativa
También se encuentra disponible un archivo `notebook.ipynb` que puede ser utilizado de forma alternativa para visualizar y ejecutar el código en un entorno de notebook.
