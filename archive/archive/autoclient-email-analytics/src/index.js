import { createClient } from "@supabase/supabase-js";
import renderHtml from "./renderHtml.js";

const RETRY_ATTEMPTS = 3;
const RETRY_DELAY = 1000; // 1 second

async function retryOperation(operation, attempts = RETRY_ATTEMPTS) {
  for (let i = 0; i < attempts; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === attempts - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
    }
  }
}

function calculateAverageTimeMetric(campaigns, endTimeField, startTimeField) {
  const validCampaigns = campaigns.filter(c => c[endTimeField] && c[startTimeField]);
  if (validCampaigns.length === 0) return 0;
  
  const totalTime = validCampaigns.reduce((sum, campaign) => {
    const endTime = new Date(campaign[endTimeField]);
    const startTime = new Date(campaign[startTimeField]);
    return sum + (endTime - startTime);
  }, 0);
  
  return Math.round(totalTime / validCampaigns.length / 1000 / 60); // Returns minutes
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Health check endpoint
    if (url.pathname === '/health') {
      return new Response('OK', { status: 200 });
    }

    const trackingId = url.searchParams.get('id');
    const trackingType = url.searchParams.get('type');
    const originalUrl = url.searchParams.get('url');

    const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_KEY);

    if (trackingId && trackingType) {
      try {
        const result = await retryOperation(async () => {
          const { data, error } = await supabase
            .from('email_campaigns')
            .select('id, open_count, click_count, status')
            .eq('tracking_id', trackingId)
            .single();

          if (error) throw error;
          if (!data) throw new Error('Campaign not found');

          const updateData = {
            status: data.status === 'sent' ? 'engaged' : data.status
          };

          if (trackingType === 'open') {
            updateData.opened_at = new Date().toISOString();
            updateData.open_count = (data.open_count || 0) + 1;
          } else if (trackingType === 'click') {
            updateData.clicked_at = new Date().toISOString();
            updateData.click_count = (data.click_count || 0) + 1;
          }

          const { error: updateError } = await supabase
            .from('email_campaigns')
            .update(updateData)
            .eq('id', data.id);

          if (updateError) throw updateError;
          return data;
        });

        if (trackingType === 'open') {
          return new Response(
            new Uint8Array([71, 73, 70, 56, 57, 97, 1, 0, 1, 0, 128, 0, 0, 255, 255, 255, 0, 0, 0, 33, 249, 4, 1, 0, 0, 0, 0, 44, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 68, 1, 0, 59]),
            { 
              headers: { 
                'Content-Type': 'image/gif',
                'Cache-Control': 'no-store, no-cache, must-revalidate',
                'Pragma': 'no-cache'
              } 
            }
          );
        } else if (trackingType === 'click' && originalUrl) {
          // Add fallback URL in case tracking fails
          const fallbackUrl = new URL(originalUrl);
          fallbackUrl.searchParams.append('tracking_failed', 'true');
          return Response.redirect(originalUrl, 302);
        }
      } catch (error) {
        console.error('Error:', error);
        if (trackingType === 'click' && originalUrl) {
          // Fallback to original URL if tracking fails
          return Response.redirect(originalUrl, 302);
        }
        return new Response('Internal Server Error', { 
          status: 500,
          headers: {
            'Content-Type': 'text/plain',
            'Cache-Control': 'no-store'
          }
        });
      }
    } else {
      // If not a tracking request, show the dashboard
      try {
        const [campaignStats, automationLogs] = await Promise.all([
          retryOperation(async () => {
            const { data, error } = await supabase
              .from('email_campaigns')
              .select(`
                id, 
                sent_at, 
                status, 
                open_count, 
                click_count,
                opened_at,
                clicked_at,
                lead_id
              `)
              .order('sent_at', { ascending: false })
              .limit(1000);

            if (error) throw error;
            return data;
          }),
          retryOperation(async () => {
            const { data, error } = await supabase
              .from('automation_logs')
              .select('logs, start_time')
              .order('start_time', { ascending: false })
              .limit(20);

            if (error) throw error;
            return data;
          })
        ]);

        const totalSent = campaignStats.length;
        const totalOpened = campaignStats.filter(ec => ec.open_count > 0).length;
        const totalClicked = campaignStats.filter(ec => ec.click_count > 0).length;
        const openRate = totalSent > 0 ? (totalOpened / totalSent * 100).toFixed(2) : 0;
        const clickRate = totalSent > 0 ? (totalClicked / totalSent * 100).toFixed(2) : 0;
        const averageTimeToOpen = calculateAverageTimeMetric(campaignStats, 'opened_at', 'sent_at');
        const averageTimeToClick = calculateAverageTimeMetric(campaignStats, 'clicked_at', 'sent_at');

        const logs = automationLogs.flatMap(log => 
          Array.isArray(log.logs) 
            ? log.logs.map(l => `${new Date(log.start_time).toLocaleString()}: ${l}`)
            : [`${new Date(log.start_time).toLocaleString()}: ${log.logs}`]
        );

        const html = await renderHtml(
          { 
            totalSent, 
            totalOpened, 
            totalClicked, 
            openRate, 
            clickRate,
            averageTimeToOpen,
            averageTimeToClick
          },
          logs
        );

        return new Response(html, {
          headers: { 
            'Content-Type': 'text/html',
            'Cache-Control': 'no-cache'
          },
        });
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        return new Response(`Error fetching dashboard data: ${error.message}`, { 
          status: 500,
          headers: {
            'Content-Type': 'text/plain',
            'Cache-Control': 'no-store'
          }
        });
      }
    }
  },
};