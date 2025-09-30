# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy project and restore dependencies first (layer caching)
COPY ["BabyLLM/BabyLLM.csproj", "BabyLLM/"]
RUN dotnet restore "BabyLLM/BabyLLM.csproj"

# Copy everything else and publish
COPY . .
WORKDIR "/src/BabyLLM"
RUN dotnet publish "BabyLLM.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# Copy published output
COPY --from=build /app/publish .

# Explicitly copy config and models folders from repo root
COPY Config ./Config
COPY models ./models

# Expose HTTP port (Kestrel defaults to 8080 inside container)
EXPOSE 8080

ENTRYPOINT ["dotnet", "BabyLLM.dll"]
