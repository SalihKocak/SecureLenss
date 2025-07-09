import random
from datetime import datetime

class RecommendationSystem:
    def __init__(self):
        """Initialize recommendation system"""
        self._recommendations = [
            {
                'title': 'URL Güvenliği',
                'description': 'HTTPS kullanımını kontrol edin ve bilinmeyen bağlantılara dikkat edin.',
                'category': 'url',
                'priority': 'high'
            },
            {
                'title': 'Email Güvenliği',
                'description': 'Şüpheli ekler ve phishing linklerine karşı dikkatli olun.',
                'category': 'email',
                'priority': 'high'
            },
            {
                'title': 'Dosya Güvenliği',
                'description': 'Bilinmeyen kaynaklardan gelen dosyaları açmadan önce tarayın.',
                'category': 'file',
                'priority': 'high'
            }
        ]
        
        self._daily_tips = [
            {
                'title': '🔒 Güçlü Şifre Kullanın',
                'description': 'En az 12 karakter, büyük-küçük harf, sayı ve özel karakter içeren şifreler kullanın.',
                'category': 'password'
            },
            {
                'title': '🔄 Yazılımları Güncel Tutun',
                'description': 'İşletim sistemi ve uygulamalarınızı düzenli olarak güncelleyin.',
                'category': 'update'
            }
        ]
        
        self._security_alerts = [
            {
                'type': 'warning',
                'title': '⚠️ Phishing Saldırı Artışı',
                'message': 'Son dönemde phishing e-postalarında artış gözlemlendi.',
                'level': 'medium'
            }
        ]
        
        self._cyber_crime_stats = [
            {
                'statistic': 'Phishing saldırıları 2023\'te %65 arttı',
                'description': 'E-posta tabanlı saldırılar en yaygın siber tehdit türü olmaya devam ediyor.'
            }
        ]

    @property
    def recommendations(self):
        """Get recommendations list"""
        return self._recommendations

    @property
    def daily_tips(self):
        """Get daily security tips"""
        return self._daily_tips

    @property
    def security_alerts(self):
        """Get security alerts"""
        return self._security_alerts

    @property
    def cyber_crime_stats(self):
        """Get cyber crime statistics"""
        return self._cyber_crime_stats

    def get_recommendations(self):
        """Return security recommendations"""
        return {
            'recommendations': self.recommendations,
            'total': len(self.recommendations),
            'timestamp': datetime.now().isoformat()
        }

    def get_daily_recommendations(self):
        """Günlük tavsiyeler ve güvenlik bilgileri getir"""
        # Günün ipucu (sabit seed ile günlük değişir)
        today = datetime.now().date()
        random.seed(today.toordinal())
        
        daily_tip = random.choice(self.daily_tips)
        security_alert = random.choice(self.security_alerts)
        cyber_stat = random.choice(self.cyber_crime_stats)
        
        # Risk seviyeleri
        current_threats = {
            'phishing': {'level': 'Yüksek', 'color': 'red', 'trend': 'artış'},
            'malware': {'level': 'Orta', 'color': 'orange', 'trend': 'sabit'},
            'scam': {'level': 'Yüksek', 'color': 'red', 'trend': 'artış'},
            'identity_theft': {'level': 'Orta', 'color': 'orange', 'trend': 'azalış'}
        }
        
        return {
            'daily_tip': daily_tip,
            'security_alert': security_alert,
            'cyber_statistic': cyber_stat,
            'current_threats': current_threats,
            'general_recommendations': self._get_general_recommendations(),
            'quick_tips': self._get_quick_tips(),
            'timestamp': datetime.now().isoformat()
        }

    def get_recommendations_by_risk(self, risk_type, risk_score):
        """Risk türü ve skoruna göre özel tavsiyeler"""
        recommendations = []
        
        if risk_type == 'url':
            recommendations = self._get_url_recommendations(risk_score)
        elif risk_type == 'email':
            recommendations = self._get_email_recommendations(risk_score)
        elif risk_type == 'file':
            recommendations = self._get_file_recommendations(risk_score)
        
        return {
            'specific_recommendations': recommendations,
            'general_tips': self._get_general_tips_by_risk(risk_score),
            'prevention_steps': self._get_prevention_steps(risk_type)
        }

    def _get_general_recommendations(self):
        """Genel güvenlik tavsiyeleri"""
        return [
            'Düzenli olarak şifrelerinizi değiştirin',
            'Bilinmeyen kaynaklardan dosya indirmeyin',
            'Sistem güncellemelerini ihmal etmeyin',
            'E-posta eklerini açmadan önce kontrol edin',
            'Güvenlik yazılımınızı güncel tutun',
            'Şüpheli aktiviteleri bildirin',
            'Sosyal medyada kişisel bilgi paylaşımında dikkatli olun'
        ]

    def _get_quick_tips(self):
        """Hızlı güvenlik ipuçları"""
        tips = [
            {'icon': '🔒', 'text': 'Şifre yöneticisi kullanın'},
            {'icon': '📧', 'text': 'E-posta adresini doğrulayın'},
            {'icon': '🔗', 'text': 'Linklerin üzerine gelip kontrol edin'},
            {'icon': '💾', 'text': 'Önemli verileri yedekleyin'},
            {'icon': '🛡️', 'text': 'Güvenlik duvarını aktif tutun'},
            {'icon': '📱', 'text': '2FA kullanın'},
            {'icon': '🌐', 'text': 'VPN kullanmayı düşünün'},
            {'icon': '⚠️', 'text': 'Şüpheli aktiviteleri rapor edin'}
        ]
        
        # Her seferinde farklı 4 ipucu döndür
        return random.sample(tips, 4)

    def _get_url_recommendations(self, risk_score):
        """URL risk skoruna göre tavsiyeler"""
        if risk_score >= 70:
            return [
                'Bu URL\'ye kesinlikle erişmeyin',
                'Bağlantıyı paylaşan kişiyi uyarın',
                'URL\'yi güvenlik ekibine bildirin',
                'Tarayıcınızda güvenlik uyarılarını dikkate alın'
            ]
        elif risk_score >= 40:
            return [
                'Bu URL\'ye dikkatli yaklaşın',
                'Kişisel bilgi girmeyin',
                'Site sertifikasını kontrol edin',
                'Şüpheli durumda sayfayı kapatın'
            ]
        else:
            return [
                'URL güvenli görünse de dikkatli olun',
                'HTTPS bağlantısını tercih edin',
                'Site adresini çift kontrol edin'
            ]

    def _get_email_recommendations(self, risk_score):
        """E-posta risk skoruna göre tavsiyeler"""
        if risk_score >= 70:
            return [
                'Bu e-posta phishing saldırısı olabilir',
                'Hiçbir linke tıklamayın',
                'E-postayı spam olarak işaretleyin',
                'Göndericiyi engelleyin',
                'BT departmanını bilgilendirin'
            ]
        elif risk_score >= 40:
            return [
                'E-posta şüpheli görünüyor',
                'Gönderen adresini doğrulayın',
                'Bağlantılara dikkatli tıklayın',
                'Ek dosyaları açmayın'
            ]
        else:
            return [
                'E-posta güvenli görünüyor',
                'Yine de standart önlemleri alın',
                'Şüphelendiğinizde doğrulama yapın'
            ]

    def _get_file_recommendations(self, risk_score):
        """Dosya risk skoruna göre tavsiyeler"""
        if risk_score >= 70:
            return [
                'Bu dosyayı kesinlikle açmayın',
                'Dosyayı hemen silin',
                'Antivirüs taraması yapın',
                'Sistem güvenliğini kontrol edin'
            ]
        elif risk_score >= 40:
            return [
                'Dosyayı açmadan önce tarayın',
                'Kaynağını doğrulayın',
                'Yedek sistem üzerinde test edin',
                'İzole ortamda kontrol edin'
            ]
        else:
            return [
                'Dosya güvenli görünüyor',
                'Yine de antivirüs taraması yapın',
                'Güvenilir kaynaktan geldiğini doğrulayın'
            ]

    def _get_general_tips_by_risk(self, risk_score):
        """Risk skoruna göre genel ipuçları"""
        if risk_score >= 70:
            return [
                'Acil güvenlik önlemleri alın',
                'Şifrelerinizi değiştirin',
                'Hesap hareketlerinizi kontrol edin',
                'Güvenlik uzmanına danışın'
            ]
        elif risk_score >= 40:
            return [
                'Ekstra dikkatli olun',
                'Güvenlik yazılımınızı güncelleyin',
                'Sistemde anormal aktivite kontrol edin'
            ]
        else:
            return [
                'Standart güvenlik uygulamalarını sürdürün',
                'Düzenli güvenlik kontrolü yapın'
            ]

    def _get_prevention_steps(self, risk_type):
        """Risk türüne göre önleme adımları"""
        prevention_steps = {
            'url': [
                'URL kısaltma servislerini dikkatli kullanın',
                'Resmi web sitelerini yer imlerine ekleyin',
                'Tarayıcı güvenlik ayarlarını aktif edin',
                'DNS filtreleme kullanın'
            ],
            'email': [
                'Spam filtreleri aktif edin',
                'E-posta istemcinizi güncel tutun',
                'Bilinmeyen gönderenlerden gelen e-postaları otomatik blokla',
                'E-posta güvenlik eğitimleri alın'
            ],
            'file': [
                'Dosya indirmek için güvenilir kaynakları kullanın',
                'Gerçek zamanlı antivirüs koruması aktif edin',
                'Dosya uzantılarını görünür yapın',
                'Sandboxing teknolojileri kullanın'
            ]
        }
        
        return prevention_steps.get(risk_type, [])

    def get_threat_intelligence(self):
        """Güncel tehdit istihbaratı"""
        return {
            'trending_threats': [
                'AI destekli phishing saldırıları',
                'Deepfake ses dolandırıcılığı',
                'QR kod tabanlı saldırılar',
                'Supply chain saldırıları'
            ],
            'vulnerable_sectors': [
                'Finans',
                'Sağlık',
                'Eğitim',
                'E-ticaret'
            ],
            'recommended_actions': [
                'Çalışan eğitimlerini artırın',
                'Incident response planını güncelleyin',
                'Güvenlik monitoring\'i güçlendirin',
                'Zero-trust yaklaşımı benimseyin'
            ]
        } 
