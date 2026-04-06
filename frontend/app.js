document.addEventListener('DOMContentLoaded', () => {
    const locateBtn = document.getElementById('locateBtn');
    const searchBtn = document.getElementById('searchBtn');
    const locationStatus = document.getElementById('locationStatus');
    const heroSection = document.getElementById('hero');
    const filtersSection = document.getElementById('filters');
    const resultsArea = document.getElementById('resultsArea');
    const cardsGrid = document.getElementById('cardsGrid');
    const loader = document.getElementById('loader');

    // Modal elements
    const placeModal = document.getElementById('placeModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const modalPhotos = document.getElementById('modalPhotos');
    const modalTitle = document.getElementById('modalTitle');
    const modalLocality = document.getElementById('modalLocality');
    const directionsBtn = document.getElementById('directionsBtn');
    const modalLoading = document.getElementById('modalLoading');
    
    // Modal Event Listeners
    closeModalBtn.addEventListener('click', closePlaceModal);
    placeModal.addEventListener('click', (e) => {
        if (e.target === placeModal) {
            closePlaceModal();
        }
    });

    function closePlaceModal() {
        placeModal.classList.add('hidden');
    }

    function openPlaceModal(place) {
        modalTitle.textContent = place.place_name || 'Unknown Place';
        modalLocality.textContent = place.locality || place.city;
        
        // Build directions url
        const destStr = encodeURIComponent(place.latitude + ',' + place.longitude);
        let dirUrl = `https://www.google.com/maps/dir/?api=1&destination=${destStr}`;
        if (place.place_id) {
            dirUrl += `&destination_place_id=${place.place_id}`;
        }
        directionsBtn.href = dirUrl;
        
        modalPhotos.innerHTML = '';
        modalLoading.classList.remove('hidden');
        placeModal.classList.remove('hidden');

        if (place.place_id && typeof google !== 'undefined' && google.maps && google.maps.places) {
            const dummyDiv = document.createElement('div');
            const service = new google.maps.places.PlacesService(dummyDiv);
            service.getDetails({
                placeId: place.place_id,
                fields: ['photos']
            }, (details, status) => {
                modalLoading.classList.add('hidden');
                if (status === google.maps.places.PlacesServiceStatus.OK && details.photos && details.photos.length > 0) {
                    const photosToFetch = Math.min(details.photos.length, 5);
                    for (let i = 0; i < photosToFetch; i++) {
                        const photoUrl = details.photos[i].getUrl({ maxWidth: 400, maxHeight: 400 });
                        const img = document.createElement('img');
                        img.src = photoUrl;
                        img.className = 'modal-photo';
                        img.alt = place.place_name;
                        modalPhotos.appendChild(img);
                    }
                } else {
                    modalPhotos.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem;">No photos available.</p>';
                }
            });
        } else {
            modalLoading.classList.add('hidden');
            modalPhotos.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem;">Photos not available.</p>';
        }
    }

    let userLat = null;
    let userLng = null;

    // Theme Toggle (Optional feature, starts in dark mode)
    const themeToggle = document.getElementById('themeToggle');
    themeToggle.addEventListener('click', () => {
        const i = themeToggle.querySelector('i');
        if (i.classList.contains('ph-moon')) {
            i.classList.replace('ph-moon', 'ph-sun');
            document.documentElement.style.setProperty('--bg-color', '#f8fafc');
            document.documentElement.style.setProperty('--text-primary', '#0f172a');
            document.documentElement.style.setProperty('--surface-color', 'rgba(0, 0, 0, 0.03)');
            document.documentElement.style.setProperty('--surface-border', 'rgba(0, 0, 0, 0.08)');
        } else {
            i.classList.replace('ph-sun', 'ph-moon');
            document.documentElement.style.setProperty('--bg-color', '#0c0f14');
            document.documentElement.style.setProperty('--text-primary', '#f8fafc');
            document.documentElement.style.setProperty('--surface-color', 'rgba(255, 255, 255, 0.03)');
            document.documentElement.style.setProperty('--surface-border', 'rgba(255, 255, 255, 0.08)');
        }
    });

    locateBtn.addEventListener('click', () => {
        if (!navigator.geolocation) {
            locationStatus.textContent = "Geolocation is not supported by your browser.";
            locationStatus.style.color = "#ef4444";
            return;
        }

        locationStatus.textContent = "Requesting location access...";
        locationStatus.style.color = "var(--text-secondary)";
        locateBtn.classList.remove('pulse-anim'); // Stop pulsing while loading
        locateBtn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Locating...';

        navigator.geolocation.getCurrentPosition(
            (position) => {
                userLat = position.coords.latitude;
                userLng = position.coords.longitude;
                locationStatus.textContent = "Location secured! Generating recommendations...";
                locationStatus.style.color = "#10b981";
                
                // Collapse hero, show filters & fetch results
                setTimeout(() => {
                    heroSection.classList.add('collapse');
                    filtersSection.classList.remove('hidden');
                    fetchRecommendations();
                }, 800);
            },
            (error) => {
                locationStatus.textContent = "Unable to retrieve your location. " + error.message;
                locationStatus.style.color = "#ef4444";
                locateBtn.classList.add('pulse-anim');
                locateBtn.innerHTML = '<i class="ph-fill ph-navigation-arrow"></i> Detect My Location';
            }
        );
    });

    const manualLocBtn = document.getElementById('manualLocBtn');
    const manualLocationInput = document.getElementById('manualLocation');

    // Initialize Google Places Autocomplete
    let autocomplete = null;
    if (typeof google !== 'undefined' && google.maps && google.maps.places) {
        autocomplete = new google.maps.places.Autocomplete(manualLocationInput, {
            types: ['geocode', 'establishment'],
        });

        autocomplete.addListener('place_changed', () => {
            const place = autocomplete.getPlace();
            if (!place.geometry || !place.geometry.location) {
                return;
            }
            userLat = place.geometry.location.lat();
            userLng = place.geometry.location.lng();
            locationStatus.textContent = `Location set to ${place.name || "selected area"}! Generating recommendations...`;
            locationStatus.style.color = "#10b981";

            setTimeout(() => {
                heroSection.classList.add('collapse');
                filtersSection.classList.remove('hidden');
                fetchRecommendations();
            }, 800);
        });
    }

    manualLocBtn.addEventListener('click', () => {
        // Fallback or trigger if user selects without clicking autocomplete
        if (userLat && userLng) {
            heroSection.classList.add('collapse');
            filtersSection.classList.remove('hidden');
            fetchRecommendations();
        } else {
            locationStatus.textContent = "Please select a valid location from the dropdown.";
            locationStatus.style.color = "#ef4444";
        }
    });

    searchBtn.addEventListener('click', () => {
        if (!userLat || !userLng) {
            alert("Please detect your location first.");
            return;
        }
        fetchRecommendations();
    });

    async function fetchRecommendations() {
        resultsArea.classList.remove('hidden');
        cardsGrid.innerHTML = '';
        loader.classList.remove('hidden');

        const cuisine = document.getElementById('cuisineFilter').value;
        const occasion = document.getElementById('occasionFilter').value;
        const time = document.getElementById('timeFilter').value;
        const budget = document.getElementById('budgetFilter').value;

        // Build URL
        const params = new URLSearchParams({
            lat: userLat,
            lng: userLng,
            radius: 15.0 // 15km default
        });
        if (cuisine) params.append('cuisine', cuisine);
        if (occasion) params.append('occasion', occasion);
        if (time) params.append('time_of_day', time);
        if (budget) params.append('budget', budget);

        try {
            const response = await fetch(`/api/recommend?${params.toString()}`);
            const data = await response.json();
            
            loader.classList.add('hidden');

            if (data.status === 'success' && data.results.length > 0) {
                renderCards(data.results);
            } else if (data.status === 'error') {
                cardsGrid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; color: #ef4444; padding: 3rem;">
                        <i class="ph ph-warning" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                        <h3>Backend Error</h3>
                        <p>${data.message}</p>
                    </div>
                `;
            } else {
                cardsGrid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); padding: 3rem;">
                        <i class="ph ph-confused" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                        <h3>No places found</h3>
                        <p>Try relaxing your filters or increasing the search radius.</p>
                    </div>
                `;
            }

        } catch (error) {
            loader.classList.add('hidden');
            cardsGrid.innerHTML = `<p style="color: #ef4444;">Error fetching recommendations: ${error.message}</p>`;
        }
    }

    function renderCards(places) {
        places.forEach((place, index) => {
            const card = document.createElement('div');
            card.className = 'card';
            card.style.animationDelay = `${index * 0.1}s`;
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => openPlaceModal(place));

            // Split categories and render tags (max 3)
            const categories = place.category ? place.category.split(',').slice(0, 3) : ['Various'];
            const tagsHTML = categories.map(c => `<span class="tag">${c.trim()}</span>`).join('');

            // Score percentage
            const matchPercent = Math.round(place.relevance_score * 100);

            card.innerHTML = `
                <div class="card-score">${matchPercent}% Match</div>
                <h3 class="card-title">${place.place_name || 'Unknown Place'}</h3>
                <div class="card-locality" style="margin-bottom: 0.25rem; font-weight: 500; color: #10b981;">
                    <i class="ph-fill ph-car"></i> 
                    ${place.travel_time ? `${place.travel_time} drive (${place.travel_dist_text})` : `${place.distance_km}km away`}
                </div>
                <div class="card-locality">
                    <i class="ph-fill ph-map-pin"></i> 
                    ${place.locality || place.city}
                </div>
                
                <div class="card-cuisine">
                    ${tagsHTML}
                </div>
                
                <div class="card-footer">
                    <div class="stat rating">
                        <i class="ph-fill ph-star"></i>
                        <span>${place.bayesian_rating.toFixed(1)} (${place.num_reviews} reviews)</span>
                    </div>
                    <div class="stat cost">
                        <i class="ph ph-wallet"></i>
                        <span>${place.price_label}</span>
                    </div>
                </div>
            `;
            cardsGrid.appendChild(card);
        });
    }
});
