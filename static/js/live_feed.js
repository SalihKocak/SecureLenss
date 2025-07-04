// Live Feed Modal JavaScript
let liveFeedInterval = null;
let autoRefreshEnabled = true;
let lastFeedItems = []; // Son feed öğelerini takip et
let newItemCount = 0; // Yeni öğe sayısı
let isLoading = false; // Yükleme durumunu takip et

// Modal açma fonksiyonu
function showLiveFeedModal() {
    // Mobile menu'yu kapat
    const mobileMenu = document.getElementById('mobileMenu');
    if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
        mobileMenu.classList.add('hidden');
    }
    
    const modal = document.getElementById('liveFeedModal');
    if (modal) {
        // Eğer modal zaten açıksa tekrar başlatma
        if (!modal.classList.contains('hidden')) {
            return;
        }
        
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        
        // Auto refresh'i aktif yap
        autoRefreshEnabled = true;
        
        // Auto refresh toggle'ını aktif yap
        const toggle = document.getElementById('autoRefreshToggle');
        if (toggle) {
            toggle.checked = true;
        }
        
        // Feed'i yükle (sadece modal ilk açıldığında)
        loadLiveFeed();
        
        // Otomatik yenileme başlat (sadece çalışmıyorsa)
        if (!liveFeedInterval) {
            startAutoRefresh();
        }
        
        console.log('🔴 CANLI FEED AKTİF - Her 10 saniyede güncelleniyor');
    }
}

// Modal kapatma fonksiyonu
function closeLiveFeedModal() {
    const modal = document.getElementById('liveFeedModal');
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
        
        // Otomatik yenilemeyi durdur
        stopAutoRefresh();
    }
}

// Feed yükleme fonksiyonu
async function loadLiveFeed() {
    // Eğer zaten yükleniyor ise bekle
    if (isLoading) {
        console.log('⏳ Feed zaten yükleniyor, bekleniyor...');
        return;
    }
    
    isLoading = true;
    showFeedState('loading');
    
    try {
        const response = await fetch('/security-feed');
        const data = await response.json();
        
        if (data.success) {
            // Güvenlik kontrolleri ekle
            const stats = data.stats || { total: 0, high_risk: 0, avg_risk: 0 };
            const feed = data.feed || [];
            const lastUpdated = data.last_updated || new Date().toISOString();
            
            updateFeedStats(stats);
            renderFeedItems(feed);
            updateLastUpdated(lastUpdated);
            
            if (feed.length === 0) {
                showFeedState('empty');
            } else {
                showFeedState('content');
            }
        } else {
            throw new Error(data.error || 'Feed yüklenemedi');
        }
    } catch (error) {
        console.error('Live feed error:', error);
        showFeedState('error');
        showNotification('Feed yüklenirken hata oluştu: ' + error.message, 'error');
    } finally {
        // Loading durumunu her zaman temizle
        isLoading = false;
    }
}

// Feed durumlarını gösterme
function showFeedState(state) {
    const states = ['loading', 'content', 'empty', 'error'];
    
    states.forEach(s => {
        const element = document.getElementById(`feed${s.charAt(0).toUpperCase() + s.slice(1)}`);
        if (element) {
            if (s === state) {
                element.classList.remove('hidden');
            } else {
                element.classList.add('hidden');
            }
        }
    });
    
    // Container'ı özel olarak kontrol et
    const container = document.getElementById('feedContainer');
    if (container) {
        if (state === 'content') {
            container.classList.remove('hidden');
        } else {
            container.classList.add('hidden');
        }
    }
}

// İstatistikleri güncelleme
function updateFeedStats(stats) {
    // Stats nesnesinin var olduğunu kontrol et
    if (!stats) {
        stats = { total: 0, total_today: 0, high_risk: 0, avg_risk: 0 };
    }
    
    const totalAllElement = document.getElementById('feedStatsTotalAll');
    const totalTodayElement = document.getElementById('feedStatsTotal');
    const highRiskElement = document.getElementById('feedStatsHighRisk');
    const avgRiskElement = document.getElementById('feedStatsAvgRisk');
    
    if (totalAllElement) totalAllElement.textContent = stats.total || 0;
    if (totalTodayElement) totalTodayElement.textContent = stats.total_today || 0;
    if (highRiskElement) highRiskElement.textContent = stats.high_risk || stats.high_risk_count || 0;
    if (avgRiskElement) avgRiskElement.textContent = stats.avg_risk || stats.avg_risk_score || 0;
}

// Feed öğelerini render etme
function renderFeedItems(feedItems) {
    const container = document.getElementById('feedContainer');
    if (!container) return;
    
    // FeedItems'ın array olduğunu kontrol et
    if (!Array.isArray(feedItems)) {
        console.warn('Feed items is not an array:', feedItems);
        feedItems = [];
    }
    
    // Yeni öğeleri tespit et
    const newItems = detectNewItems(feedItems);
    
    container.innerHTML = '';
    
    feedItems.forEach((item, index) => {
        if (item && typeof item === 'object') {
            const feedItemElement = createFeedItemElement(item);
            
            // Yeni öğeyi highlight et
            if (newItems.includes(item.id)) {
                feedItemElement.classList.add('new-item-highlight');
                
                // 3 saniye sonra highlight'ı kaldır
                setTimeout(() => {
                    feedItemElement.classList.remove('new-item-highlight');
                    feedItemElement.classList.add('new-item-fade');
                }, 3000);
            }
            
            container.appendChild(feedItemElement);
        }
    });
    
    // Son feed öğelerini güncelle
    lastFeedItems = feedItems.map(item => item.id);
    
    // Yeni öğe sayısını göster
    if (newItems.length > 0) {
        showNewItemNotification(newItems.length);
    }
}

// Yeni öğeleri tespit et
function detectNewItems(currentItems) {
    if (lastFeedItems.length === 0) {
        return []; // İlk yükleme, hiçbiri yeni değil
    }
    
    const currentIds = currentItems.map(item => item.id);
    const newItemIds = currentIds.filter(id => !lastFeedItems.includes(id));
    
    return newItemIds;
}

// Yeni öğe bildirimi göster
function showNewItemNotification(count) {
    const message = count === 1 ? 
        '🔔 Yeni analiz sonucu geldi!' : 
        `🔔 ${count} yeni analiz sonucu geldi!`;
    
    showNotification(message, 'success', 3000);
    newItemCount += count;
}

// Feed öğesi elementi oluşturma
function createFeedItemElement(item) {
    const div = document.createElement('div');
    div.className = 'feed-item-modern feed-item-loading';
    div.setAttribute('data-risk', item.severity);
    
    const timeAgo = formatAnalysisDateTime(item.timestamp);
    const iconClass = getAnalysisIcon(item.type);
    const typeLabel = getTypeLabel(item.type);
    const maskedContent = maskSensitiveContent(item.query_preview);
    const riskIcon = getRiskIcon(item.severity);
    
    div.innerHTML = `
        <div class="feed-card">
            <!-- Header Row -->
            <div class="feed-header">
                <div class="feed-type-section">
                    <div class="feed-type-icon ${item.color === 'red' ? 'icon-red' : 
                                                item.color === 'orange' ? 'icon-orange' : 
                                                item.color === 'yellow' ? 'icon-yellow' : 
                                                item.color === 'green' ? 'icon-green' : 'icon-blue'}">
                        <i class="${iconClass}"></i>
                    </div>
                    <div class="feed-type-info">
                        <span class="feed-type-label">${typeLabel} Analizi</span>
                        <span class="feed-time-stamp">${timeAgo}</span>
                    </div>
                </div>
                <div class="feed-status-badge ${item.severity}">
                    <i class="${riskIcon}"></i>
                    <span>${item.risk_level}</span>
                </div>
            </div>

            <!-- Content Row -->
            <div class="feed-content">
                <div class="feed-content-header">
                    <i class="fas fa-eye"></i>
                    <span>Analiz Edilen İçerik</span>
                </div>
                <div class="feed-content-preview">
                    ${maskedContent}
                </div>
            </div>

            <!-- Footer Row -->
            <div class="feed-footer">
                <div class="feed-risk-info">
                    <span class="risk-score-badge">
                        <i class="fas fa-percentage"></i>
                        ${item.risk_score}% Risk
                    </span>
                </div>
                <div class="feed-meta">
                    <span class="feed-timestamp">
                        <i class="fas fa-clock"></i>
                        ${getTurkeyTime(item.timestamp)}
                    </span>
                </div>
            </div>
        </div>
    `;
    
    return div;
}

// İçeriği maskeleme fonksiyonu
function maskSensitiveContent(content) {
    if (!content || content.length === 0) {
        return '***** ***** *****';
    }
    
    // İçeriği kelimelerine ayır
    const words = content.split(' ');
    const maskedWords = [];
    
    for (let i = 0; i < words.length; i++) {
        const word = words[i];
        if (word.length <= 3) {
            // Kısa kelimeleri tamamen maskele
            maskedWords.push('*'.repeat(word.length));
        } else if (i === 0 || i === words.length - 1) {
            // İlk ve son kelimeyi kısmen göster
            maskedWords.push(word.charAt(0) + '*'.repeat(word.length - 2) + word.charAt(word.length - 1));
        } else {
            // Ortadaki kelimeleri tamamen maskele
            maskedWords.push('*'.repeat(Math.min(word.length, 5)));
        }
    }
    
    // Maksimum 50 karakter göster
    const masked = maskedWords.join(' ');
    if (masked.length > 50) {
        return masked.substring(0, 47) + '...';
    }
    
    return masked;
}

// Risk seviyesi ikonu
function getRiskIcon(severity) {
    switch (severity) {
        case 'critical': return 'fas fa-exclamation-triangle';
        case 'high': return 'fas fa-exclamation-circle';
        case 'medium': return 'fas fa-info-circle';
        case 'low': return 'fas fa-check-circle';
        case 'safe': return 'fas fa-shield-check';
        default: return 'fas fa-question-circle';
    }
}

// Analiz durumu rozeti
function getAnalysisStatusBadge(riskScore) {
    if (riskScore >= 80) {
        return '<span class="px-2 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium"><i class="fas fa-exclamation-triangle mr-1"></i>Yüksek Risk</span>';
    } else if (riskScore >= 60) {
        return '<span class="px-2 py-1 bg-orange-100 text-orange-700 rounded-full text-xs font-medium"><i class="fas fa-exclamation-circle mr-1"></i>Orta Risk</span>';
    } else if (riskScore >= 30) {
        return '<span class="px-2 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs font-medium"><i class="fas fa-info-circle mr-1"></i>Düşük Risk</span>';
    } else {
        return '<span class="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium"><i class="fas fa-shield-check mr-1"></i>Güvenli</span>';
    }
}

// Analiz tipi ikonu
function getAnalysisIcon(type) {
    switch (type) {
        case 'url': return 'fas fa-link';
        case 'email': return 'fas fa-envelope';
        case 'file': return 'fas fa-file';
        default: return 'fas fa-question';
    }
}

// Tip etiketi
function getTypeLabel(type) {
    switch (type) {
        case 'url': return 'URL';
        case 'email': return 'E-posta';
        case 'file': return 'Dosya';
        default: return 'Bilinmeyen';
    }
}

// Türkiye saatini formatla
function getTurkeyTime(timestamp) {
    // UTC timestamp'i parse et
    let utcTime;
    if (typeof timestamp === 'string') {
        // Eğer Z ile bitmiyorsa UTC olarak işaretle
        utcTime = new Date(timestamp + (timestamp.endsWith('Z') ? '' : 'Z'));
    } else {
        utcTime = new Date(timestamp);
    }
    
    // Türkiye timezone'una çevir (UTC+3)
    const turkeyTime = new Date(utcTime.getTime() + (3 * 60 * 60 * 1000));
    
    // Türkiye formatında saat:dakika formatı
    return turkeyTime.toLocaleTimeString('tr-TR', {
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'UTC' // UTC olarak işle çünkü manuel offset ekledik
    });
}

// Analiz tarih-saatini formatla (26/06/25 16:59 formatında)
function formatAnalysisDateTime(timestamp) {
    // UTC timestamp'i parse et
    let utcTime;
    if (typeof timestamp === 'string') {
        utcTime = new Date(timestamp + (timestamp.endsWith('Z') ? '' : 'Z'));
    } else {
        utcTime = new Date(timestamp);
    }
    
    // Türkiye timezone'una çevir (UTC+3)
    const turkeyTime = new Date(utcTime.getTime() + (3 * 60 * 60 * 1000));
    
    // Şu an Türkiye saati
    const now = new Date(Date.now() + (3 * 60 * 60 * 1000));
    const diffMs = now.getTime() - turkeyTime.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    // Zaman farkını Türkçe olarak formatla
    if (diffMins < 1) {
        return 'şimdi';
    } else if (diffMins < 60) {
        return `${diffMins} dk önce`;
    } else if (diffHours < 24) {
        return `${diffHours} saat önce`;
    } else if (diffDays < 7) {
        return `${diffDays} gün önce`;
    } else {
        // Eski tarihler için tarih formatı
        return turkeyTime.toLocaleDateString('tr-TR', {
            day: '2-digit',
            month: '2-digit',
            year: '2-digit',
            timeZone: 'UTC'
        });
    }
}

// Zaman farkı hesaplama - Türkiye timezone fix (ihtiyaç durumunda)
function getTimeAgo(timestamp) {
    const now = new Date();
    
    // UTC timestamp'i Türkiye saatine çevir
    let time;
    if (typeof timestamp === 'string') {
        // Backend'den gelen UTC timestamp
        time = new Date(timestamp + (timestamp.endsWith('Z') ? '' : 'Z'));
    } else {
        time = new Date(timestamp);
    }
    
    // Türkiye timezone offset'i uygula (UTC+3)
    const turkeyOffset = 3 * 60 * 60 * 1000; // 3 saat milisaniye
    const localTime = new Date(time.getTime() + turkeyOffset);
    
    const diffInSeconds = Math.floor((now - localTime) / 1000);
    
    // Negatif fark varsa (gelecek zaman) "şimdi" de
    if (diffInSeconds < 0) {
        return "şimdi";
    }
    
    if (diffInSeconds < 60) {
        return `${diffInSeconds} saniye önce`;
    } else if (diffInSeconds < 3600) {
        const minutes = Math.floor(diffInSeconds / 60);
        return `${minutes} dakika önce`;
    } else if (diffInSeconds < 86400) {
        const hours = Math.floor(diffInSeconds / 3600);
        return `${hours} saat önce`;
    } else {
        const days = Math.floor(diffInSeconds / 86400);
        return `${days} gün önce`;
    }
}

// updateLastUpdated fonksiyonu - geçici fix
function updateLastUpdated(timestamp) {
    // Bu fonksiyon artık bir şey yapmıyor - gereksiz ama hata vermemesi için bırakıldı
    console.log('updateLastUpdated çağrıldı ama hiçbir şey yapmıyor (kaldırıldı)');
}

// Feed yenileme
async function refreshLiveFeed() {
    if (isLoading) {
        showNotification('Feed zaten yükleniyor, lütfen bekleyin...', 'warning', 2000);
        return;
    }
    
    await loadLiveFeed();
    showNotification('Feed yenilendi', 'success', 2000);
}

// Otomatik yenileme başlat
function startAutoRefresh() {
    if (liveFeedInterval) {
        clearInterval(liveFeedInterval);
    }
    
    if (autoRefreshEnabled) {
        // Her 25 saniyede bir yenile (kullanıcı isteği)
        liveFeedInterval = setInterval(() => {
            if (autoRefreshEnabled) {
                console.log('🔄 Otomatik feed yenileme: ' + new Date().toLocaleTimeString('tr-TR'));
                loadLiveFeed();
            }
        }, 25000); // 25 saniye = 25000ms
        
        console.log('✅ Otomatik yenileme başlatıldı (25 saniye)');
    }
}

// Otomatik yenileme durdurma
function stopAutoRefresh() {
    if (liveFeedInterval) {
        clearInterval(liveFeedInterval);
        liveFeedInterval = null;
    }
}

// Otomatik yenileme toggle
function toggleAutoRefresh() {
    const toggle = document.getElementById('autoRefreshToggle');
    if (toggle) {
        autoRefreshEnabled = toggle.checked;
        
        if (autoRefreshEnabled) {
            startAutoRefresh();
            showNotification('Otomatik yenileme açıldı', 'info', 2000);
        } else {
            stopAutoRefresh();
            showNotification('Otomatik yenileme kapatıldı', 'info', 2000);
        }
    }
}

// clearAllAnalyses fonksiyonu kaldırıldı - güvenlik riski

// Modal dışına tıklama ile kapatma
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('liveFeedModal');
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeLiveFeedModal();
            }
        });
    }
    
    // Otomatik yenileme toggle event listener
    const autoRefreshToggle = document.getElementById('autoRefreshToggle');
    if (autoRefreshToggle) {
        autoRefreshToggle.addEventListener('change', toggleAutoRefresh);
    }
    
    // ESC tuşu ile kapatma
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeLiveFeedModal();
        }
    });
    
    // Listen for live feed update requests from other modules
    document.addEventListener('liveFeedUpdateRequested', function(e) {
        console.log('🔔 Live feed update requested:', e.detail);
        if (!isLoading) {
            loadLiveFeed();
        }
    });
});

// Sayfa kapatılırken interval'ı temizle
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
});
